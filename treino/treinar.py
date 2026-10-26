"""Retreino do robô BTC (tendência + ML, horizonte de 14 dias).

Reproduz as mesmas 20 features do nó "Calcular sinal" do n8n, treina o gradient boosting e
exporta as árvores no formato que o n8n lê (modelo_btc.json).

Trava de qualidade: antes de trocar o modelo, o mesmo procedimento é treinado só com dados
anteriores aos últimos 365 dias e testado nesses 365 dias, que ele não viu. O modelo novo só
é gravado se o ML ajudar de verdade nesse período (AUC acima de 0,5 e resultado da estratégia
pelo menos igual ao da tendência sozinha).

Uso:
  python treino/treinar.py              treina, testa e grava modelo_btc.json se aprovado
  python treino/treinar.py --dry-run    só treina e testa, sem gravar
"""
import datetime as dt
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURES = ['r1', 'r3', 'r7', 'r14', 'r30', 'r60', 'r90', 'r180', 'r365', 'vol7', 'vol30', 'vol90',
            'dma20', 'dma50', 'dma100', 'dma200', 'dd_ath', 'vix', 'vix_ch', 'ens']
HORIZONTE = 14
THR_ON, THR_OFF = 0.55, 0.45
INICIO_TREINO = '2018-08-01'
PARAMS = dict(max_iter=150, max_leaf_nodes=8, learning_rate=0.05, random_state=0)
TAXA = 0.001
UA = {'User-Agent': 'Mozilla/5.0'}


def baixar_btc():
    out, inicio = [], 1502928000000
    while True:
        url = ('https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d'
               f'&limit=1000&startTime={inicio}')
        lote = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        if not lote:
            break
        out += lote
        inicio = lote[-1][0] + 1
        if len(lote) < 1000:
            break
        time.sleep(0.2)
    agora = time.time() * 1000
    out = [k for k in out if k[6] < agora]  # só candles fechados
    idx = pd.to_datetime([k[0] for k in out], unit='ms').normalize()
    return pd.Series([float(k[4]) for k in out], index=idx)


def baixar_vix():
    url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
    v = pd.read_csv(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
    v.columns = [c.strip().upper() for c in v.columns]
    v.index = pd.to_datetime(v['DATE'], format='%m/%d/%Y')
    return v['CLOSE'].astype(float).sort_index()


def montar_features(btc, vix):
    c = btc.values
    n = len(c)
    lr = np.r_[np.nan, np.diff(np.log(c))]
    s = pd.Series(c)
    ma = {w: s.rolling(w).mean().values for w in (5, 10, 20, 50, 100, 150, 200)}
    ens = np.zeros(n)
    for a in (5, 10, 20):
        for b in (50, 100, 150, 200):
            ens += ma[a] > ma[b]
    ens /= 12
    vd, vv = vix.index.values, vix.values

    def vix_lag(datas):  # pregão anterior ao último pregão <= data (igual ao n8n)
        i = np.searchsorted(vd, datas.values, side='right') - 2
        return np.where(i >= 0, vv[np.clip(i, 0, None)], np.nan)

    dias = btc.index
    vx, vx5 = vix_lag(dias), vix_lag(dias - pd.Timedelta(days=5))
    f = {}
    for k in (1, 3, 7, 14, 30, 60, 90, 180, 365):
        f[f'r{k}'] = np.log(c) - np.log(np.r_[np.full(k, np.nan), c[:-k]])
    for w in (7, 30, 90):
        f[f'vol{w}'] = pd.Series(lr).rolling(w).std().values
    for w in (20, 50, 100, 200):
        f[f'dma{w}'] = c / ma[w] - 1
    f['dd_ath'] = c / np.maximum.accumulate(c) - 1
    f['vix'], f['vix_ch'], f['ens'] = vx, vx - vx5, ens
    X = pd.DataFrame(f, index=dias)[FEATURES]
    y = pd.Series(np.r_[(c[HORIZONTE:] > c[:-HORIZONTE]).astype(float), np.full(HORIZONTE, np.nan)], index=dias)
    return X, y


def treinar(X, y, ate):
    m = (X.index + pd.Timedelta(days=HORIZONTE) < ate) & y.notna() & (X.index >= INICIO_TREINO)
    clf = HistGradientBoostingClassifier(**PARAMS)
    clf.fit(X[m].values, y[m].values)
    return clf


def exportar(clf, ath, treinado_ate):
    arvores = []
    for (pred,) in clf._predictors:
        nd = pred.nodes
        folha = nd['is_leaf'].astype(bool)
        arvores.append({
            'f': [-1 if folha[i] else int(nd['feature_idx'][i]) for i in range(len(nd))],
            't': [0.0 if folha[i] else float(nd['num_threshold'][i]) for i in range(len(nd))],
            'l': [0 if folha[i] else int(nd['left'][i]) for i in range(len(nd))],
            'r': [0 if folha[i] else int(nd['right'][i]) for i in range(len(nd))],
            'v': [float(x) for x in nd['value']],
            'm': [int(x) for x in nd['missing_go_to_left']],
        })
    return {'versao': f'tend_ml14_055_{treinado_ate}', 'treinado_ate': treinado_ate, 'features': FEATURES,
            'base': float(np.ravel(clf._baseline_prediction)[0]), 'trees': arvores, 'ath': float(ath),
            'thr_on': THR_ON, 'thr_off': THR_OFF}


def prob_json(modelo, x):
    """Mesma lógica do nó do n8n, para conferir que o JSON exportado é fiel ao modelo."""
    s = modelo['base']
    for t in modelo['trees']:
        j = 0
        while t['f'][j] != -1:
            v = x[t['f'][j]]
            if np.isnan(v):
                j = t['l'][j] if t['m'][j] else t['r'][j]
            else:
                j = t['l'][j] if v <= t['t'][j] else t['r'][j]
        s += t['v'][j]
    return 1 / (1 + np.exp(-s))


def estrategia(btc, prob, ens, ini):
    """Retorno (em US$) da regra do robô e da tendência sozinha, a partir de ini."""
    dias = btc.index[btc.index >= ini]
    ml, res = 0, {}
    alvo_robo, alvo_tend = {}, {}
    for d in dias:
        p = prob.get(d, np.nan)
        if not np.isnan(p):
            if not ml and p > THR_ON:
                ml = 1
            elif ml and p < THR_OFF:
                ml = 0
        e = 1 if ens[d] > 0.5 else 0
        alvo_robo[d], alvo_tend[d] = (1 if (e or ml) else 0), e
    for nome, alvo in (('robo', alvo_robo), ('tendencia', alvo_tend)):
        v, pos = 1.0, 0
        for d0, d1 in zip(dias, dias[1:]):
            if alvo[d0] != pos:
                v *= 1 - TAXA
                pos = alvo[d0]
            if pos:
                v *= btc[d1] / btc[d0]
        res[nome] = v - 1
    return res


def main():
    dry = '--dry-run' in sys.argv
    btc, vix = baixar_btc(), baixar_vix()
    X, y = montar_features(btc, vix)
    hoje = btc.index[-1] + pd.Timedelta(days=1)
    corte = hoje - pd.Timedelta(days=365 + HORIZONTE)

    # 1) trava: mesmo procedimento, treinado sem os últimos 365 dias e testado neles
    val = treinar(X, y, corte)
    m = (X.index >= corte) & y.notna()
    p_val = pd.Series(val.predict_proba(X[m].values)[:, 1], index=X.index[m])
    auc = roc_auc_score(y[m], p_val)
    ens = X['ens']
    r = estrategia(btc, p_val, ens, corte)
    aprovado = auc > 0.5 and r['robo'] >= r['tendencia']

    # 2) modelo final com todos os dados
    final = treinar(X, y, hoje)
    treinado_ate = (hoje - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    modelo = exportar(final, btc.max(), treinado_ate)

    # 3) o JSON precisa reproduzir o modelo exatamente
    amostra = X.dropna().iloc[-300:]
    p_sk = final.predict_proba(amostra.values)[:, 1]
    p_js = np.array([prob_json(modelo, x) for x in amostra.values])
    paridade = float(np.max(np.abs(p_sk - p_js)))
    assert paridade < 1e-9, f'JSON exportado diverge do modelo ({paridade})'

    resumo = (f'Retreino robô BTC ({treinado_ate})\n'
              f'Teste nos últimos 365 dias (fora do treino): AUC {auc:.3f} | '
              f'robô {r["robo"]*100:+.1f}% vs tendência {r["tendencia"]*100:+.1f}%\n'
              f'Resultado: {"APROVADO, modelo trocado" if aprovado and not dry else "APROVADO (dry-run, nada gravado)" if aprovado else "REPROVADO, modelo atual mantido"}')
    print(resumo)
    with open('resumo_retreino.txt', 'w') as f:
        f.write(resumo + '\n')
    if aprovado and not dry:
        with open('modelo_btc.json', 'w') as f:
            json.dump(modelo, f, separators=(',', ':'))


if __name__ == '__main__':
    main()

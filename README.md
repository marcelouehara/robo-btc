# robo-btc

Modelo usado pelo robô de alerta diário de BTC (n8n): tendência + gradient boosting, horizonte de 14 dias.

- `modelo_btc.json`: 150 árvores, treinado até 2026-10-11, versão tend_ml14_055_2026-10-11.
- O n8n baixa este arquivo todo dia às 21h05 pela URL raw do GitHub.
- Para atualizar: retreine, substitua o `modelo_btc.json` e faça commit. O n8n pega a versão nova na execução seguinte.

## Retreino automático

- `treino/treinar.py` reproduz as features do n8n, treina o modelo e exporta as árvores no formato do robô.
- O GitHub Action `Retreino anual do modelo` roda todo 2 de janeiro e também pode ser disparado à mão (aba Actions, opção "Só testar" ligada por padrão).
- **Trava:** o procedimento é treinado sem os últimos 365 dias e testado neles. O modelo só é trocado se o ML tiver AUC acima de 0,5 e a estratégia render pelo menos o mesmo que a tendência sozinha. Senão, o modelo atual é mantido.
- O resultado chega no Telegram se os secrets `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` estiverem configurados no repositório.
- Por que anual: no teste walk-forward de 2023 a 2026, retreinar todo mês, semana ou dia rendeu menos que retreinar uma vez por ano.

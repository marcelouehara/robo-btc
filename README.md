# robo-btc

Modelo usado pelo robô de alerta diário de BTC (n8n): tendência + gradient boosting, horizonte de 14 dias.

- `modelo_btc.json`: 150 árvores, treinado até 2026-10-11, versão tend_ml14_055_2026-10-11.
- O n8n baixa este arquivo todo dia às 21h05 pela URL raw do GitHub.
- Para atualizar: retreine, substitua o `modelo_btc.json` e faça commit. O n8n pega a versão nova na execução seguinte.

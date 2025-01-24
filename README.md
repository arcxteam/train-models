# Allora Worker x Reputer for (Allora Model Forge Competition)

> [!NOTE]
> **Following ANN, THE ENDED** the competition run worker in here [Dashboard](https://app.allora.network?ref=eyJyZWZlcnJlcl9pZCI6IjhmZmQ5YTMwLWZhYmMtNDJjYy05NmNiLWZmMTYxOWE3ZDI0NyJ9) has **ENDED**. Earn Points Participants can earn Allora Points through a variety of on-chain and off-chain activities

![Confirm](https://img.shields.io/badge/WHATS_THE_NEXT-ALLORA_FORGE_ONGOING-brightgreen)

# A. Allora Model Forge

Let’s forge the future together here [Connect Wallet -> Apply -> Eligible Dahsboard](https://forge.allora.network)

> The Allora Model Forge is the hub for monetization of machine learning models. Compete alongside top talent in AI, create models with real impact, and earn rewards & recognition within the Allora ecosystem and beyond.

`⚒️ 12h ETH/USD Volatility Prediction`
`⚒️ 12h ETH/USDC Volume Prediction`
`⚒️ 5 min ETH/USD Price Prediction`

![image](https://github.com/user-attachments/assets/27e75675-83dd-4bfc-ac0c-ec0be6d1ed9f)
![image](https://github.com/user-attachments/assets/5e7df15b-afcc-46a5-9031-a404b22c2d7b)


## 1. Cloning this repository

```bash
git clone https://github.com/arcxteam/allora-cpu.git
cd allora-cpu
```

## 2. Install Docker & libraries (optional if not yet installed)

```bash
chmod +x init.sh
./init.sh
```

## 3. Setup Preparation
- Run the worker - directory
- Setting wallet & phrase or direct to manual `config.json`
- Setting docker-compose.yaml
- Setting docker-compose-reputer.yaml
- Setting coingecko API=`CG-xxxxx`

```bash
cd allora-cpu/allora-node
```

```bash
chmod +x ./init.config.sh
./init.config.sh "wallet_name" "mnemonic_phrase" "coingecko_api_key"
```
```diff
+ example:
- chmod +x ./init.config.sh
- ./init.config.sh "bangsat_kau" "kalo_pake_model_sendiri_pasti_dapat_point" "CG-4z765aZSHGD1"
```

## 4. Running Build Worker
```bash
docker compose pull
```
```bash
docker compose up --build -d 
```

- Check logs after run
```bash
docker compose logs -f 
```
---
<p align="center">
  <samp><img src="https://img.shields.io/badge/CONGRATS_THE_SETUP_IS_COMPLETED_-8a2be2"/><p align="center">
</p>

---

# B. Running Own Machine Learning Model

## 1. Now if you want to have your own unique models

- Trying editing
- Buidl python in the `train_models.py` file
- To edit for the best your 'own model' and run cmd
- Save the model `CTRL + X and then Y` save ENTER

```bash
nano train_models.py 
```

## 2. How to train the model?

- Run command below!!
```bash
chmod +x ./start-train-model.sh
./start-train-model.sh
```

## 3. How to check trainng the model?

- Run command below!!
- Copied of CONTAINER-ID
- docker logs -f `xxxCONTAINER-IDxxxxx`
- Enter and check the logs
  
```bash
sudo docker ps -a
```

```bash
docker logs -f 
```

## 4. Result for own training model

`Training the model in ranges 30-60 Minutes!! plz keep until success`

- Check your IMAGES run as 'allora-train-model:1.0.0'
- Check the file folder like ETH/BTC etc
- Check logs for training horizon symbol
- 5-10 Minutues & 20 Minutes & 1440 Minutes (5m-10m-20m-1D)
- The Container-ID with train will be auto exited after the completed

![Capture555554-09-30-2024_05_07_PM](https://github.com/user-attachments/assets/f415427e-a8f4-49cd-8d50-60a9df5b7113)


###### Thanks to hiephtdev & 0xtnpxsgt

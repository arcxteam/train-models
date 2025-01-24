## Allora Worker x Reputer for Own Model (Allora Model Forge Competition)

> [!NOTE]
> **THIS END** Following ann, the competition run worker in here [Dashboard](https://app.allora.network?ref=eyJyZWZlcnJlcl9pZCI6IjhmZmQ5YTMwLWZhYmMtNDJjYy05NmNiLWZmMTYxOWE3ZDI0NyJ9) has END. Earn Points Participants can earn Allora Points through a variety of on-chain and off-chain activities

![image](https://github.com/user-attachments/assets/27e75675-83dd-4bfc-ac0c-ec0be6d1ed9f)

![image](https://github.com/user-attachments/assets/5e7df15b-afcc-46a5-9031-a404b22c2d7b)


## 1. Cloning code from this repository

```bash
git clone https://github.com/arcxteam/allora-cpu.git
cd allora-cpu
```

## 2. Install Docker and necessary libraries (optional)

```bash
chmod +x init.sh
./init.sh
```

## 3. Setup Preparation
- Run the worker - directory
- Setting wallet & phrase or manual config.json
- Setting docker-compose.yaml & docker-compose-reputer.yaml =>> Coingecko API=CG-xxxxx

```bash
cd allora-cpu/allora-node
```

```bash
chmod +x ./init.config.sh
./init.config.sh "wallet_name" "mnemonic" "coingecko_api_key"
```
- example: chmod +x ./init.config.sh
./init.config.sh "bangsat-kau" "kalo pake model sendiri pasti dapat point xxx xxx" "CG-xxxxxx"

## 4. Upgrade & Build Your Worker
```bash
docker compose pull
```
```bash
docker compose up --build -d 
```

- Check logs as to running
```bash
docker compose logs -f 
```

---------------------------------------- Congrats Your Setup is Completed -------------------------------------


## 1. Now if you want to have your own unique model

- Run command below!!
- Buidl with the train_models.py file
- To edit for the best your 'own model' and run
- Save the model (ctrl X + Y) save ENTER

```bash
nano train_models.py 
```

## 2. How to train the model?

- Run command below!!
```bash
chmod +x ./start-train-model.sh
./start-train-model.sh
```

## 3. How to check train the model?

- Run docker command!!
- Copied of CONTAINER ID
- docker logs -f xxxxxxxxxx
- Enter and check the logs
  
```bash
sudo docker ps
```

```bash
docker logs -f 
```

## 4. Training the model in ranges 30-60 Minutes!! plz keep until success

- Check your IMAGES run as 'allora-train-model:1.0.0'
- Check logs for training horizon symbol
- 10 Minutues & 20 Minutes & 1440 Minutes
- The containerID with train will be exited auto after complete 

![Capture555554-09-30-2024_05_07_PM](https://github.com/user-attachments/assets/f415427e-a8f4-49cd-8d50-60a9df5b7113)




###### Thanks to hiephtdev & 0xtnpxsgt

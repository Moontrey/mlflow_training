to enter mlflow server -> 
```
http://0.0.0.0:5000/	
```
to run via docker compose
```
docker-compose run model_service bash 
```

to build 
```
docker-compose build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g)
```

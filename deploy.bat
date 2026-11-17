@echo off
set "REG=226420184359.dkr.ecr.ap-south-1.amazonaws.com"
docker build --provenance=false -t clausecite . || exit /b 1
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin %REG% || exit /b 1
docker tag clausecite:latest %REG%/clausecite:latest || exit /b 1
docker push %REG%/clausecite:latest || exit /b 1
aws lambda update-function-code --function-name clausecite --image-uri %REG%/clausecite:latest --region ap-south-1
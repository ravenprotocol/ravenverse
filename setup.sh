# Remove all files
sudo rm -rf files
sudo mkdir files

#Remove images and containers
sudo docker container stop raven_server
sudo docker container rm raven_server
sudo docker rmi raven

# Create an image, create a container and start it
sudo docker build -t raven .
sudo docker run --name raven_server -p 9999:9999 raven

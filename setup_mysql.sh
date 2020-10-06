docker container stop rwd_mysql
docker container rm rwd_mysql
docker run --name=rwd_mysql --env="MYSQL_ROOT_PASSWORD=" -p 3306:3306 -d mysql:latest

sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb https://debian.neo4j.com stable 4.1"
sudo apt install neo4j
sudo systemctl enable neo4j.service
sudo systemctl status neo4j.service







# docker run -it --rm \
#   --publish=7474:7474 --publish=7687:7687 \
#   -v $HOME/bloom.license:/licenses/bloom.license \
#   --env NEO4J_AUTH=neo4j/test \
#   --env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
#   --env NEO4JLABS_PLUGINS='["bloom"]' \
#   neo4j:community

rm -rf ravcom
rm -rf ravop
rm -rf ravml
rm -rf ravjs
rm -rf ravsock
rm -rf ravviz

rm -rf repos
mkdir repos
cd repos

git clone https://github.com/ravenprotocol/ravcom.git
cp -R ravcom/ravcom ../
rm -rf ravcom

git clone https://github.com/ravenprotocol/ravop.git
cp -R ravop/ravop ../
rm -rf ravop

git clone https://github.com/ravenprotocol/ravml.git
cp -R ravml/ravml ../
rm -rf ravml

git clone https://github.com/ravenprotocol/ravsock.git
cp -R ravsock/ravsock ../
rm -rf ravsock

git clone https://github.com/ravenprotocol/ravjs.git
cp -R ravjs ../
rm -rf ravjs

git clone https://github.com/ravenprotocol/ravviz.git
cp -R ravviz ../
rm -rf ravviz

cd ../
rm -rf repos

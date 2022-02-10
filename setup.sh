rm -rf ravsock
rm -rf ravpy
rm -rf ravftp

rm -rf repos
mkdir repos
cd repos

git clone https://github.com/ravenprotocol/ravop.git
pip install -r ravop/requirements.txt
python ravop/setup.py install
cp -R ravop/ravop ../
rm -rf ravop

git clone https://github.com/ravenprotocol/ravsock.git
pip install -r ravsock/requirements.txt
cp -R ravsock/ravsock ../
rm -rf ravsock

git clone https://github.com/ravenprotocol/ravjs.git
cp -R ravjs ../
rm -rf ravjs

git clone https://github.com/ravenprotocol/ravpy.git
pip install -r ravpy/requirements.txt
cp -R ravpy/ravpy ../
rm -rf ravpy

git clone https://github.com/ravenprotocol/ravftp.git
pip install -r ravftp/requirements.txt
cp -R ravftp/ravftp ../
rm -rf ravftp


cd ../
rm -rf repos

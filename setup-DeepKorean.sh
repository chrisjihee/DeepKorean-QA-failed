# create
mkdir -p /dat/proj/DeepKorean; cd /dat/proj/DeepKorean; rm .* * -rf;
conda create --name DeepKorean python=3.10 -y; conda activate DeepKorean;

# clone
mkdir lib; cd lib;
git clone git@github.com:bcj/AttrDict.git attrdict-2.0;
git clone git@github.com:microsoft/DeepSpeed.git deepspeed-0.6;
git clone git@github.com:huggingface/transformers.git transformers-4.18;
git clone git@github.com:PyTorchLightning/pytorch-lightning.git pytorch-lightning-1.6;
cd attrdict-2.0; rm -rf .* C* M* requirements-* t*; cd ..;
cd deepspeed-0.6; rm -rf .* C* D* M* S* a* c* do* rel* t*; cd ..;
cd transformers-4.18; rm -rf .* C* I* M* README_* c* d* e* h* m* n* p* scr* t* u* v*; cd ..;
cd pytorch-lightning-1.6; rm -rf .* _* C* M* S* d* e* l* t* pyp* pl_*; chmod 664 *.*; cd ..;

# install
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y;
pip install -e attrdict-2.0; pip install -e deepspeed-0.6;
pip install -e pytorch-lightning-1.6; pip install -e transformers-4.18;
pip install --upgrade datasets flatbuffers openpyxl scikit-learn SQLAlchemy;
cd ..;

# link & copy
ln -s ../PretrainedLM pretrained;
cp ../setup-DeepKorean.sh ../.gitignore .;

# commit & push
git init; git add .git* *; git commit -m "DeepKorean with pytorch";
git remote add origin git@github.com:chrisjihee/DeepKorean.git; git push --set-upstream origin master;

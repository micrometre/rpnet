.PHONY: run install clean setup-dataset
install:
	pip3 install -r requirements.txt

run_inference:
	 python inference.py -i demo/ -m fh02.pth --cpu -o outputs
clean:
	rm -rf output outputs 
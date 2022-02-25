import base64 
import sys 

def main():
    # $ python encodeb64.py data.csv out.bin 
    input = sys.argv[1]    
    output = sys.argv[2]    
    data = open(input, "r").read()
    data_bytes = data.encode("utf-8")
    encoded = base64.b64encode(data_bytes)
    f = open(output, "wb")
    f.write(encoded)

main()
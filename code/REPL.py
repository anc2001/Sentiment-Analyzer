    
def main():
    try:
        while True:
            try:
                _in = input(">> ")
                try:
                    print(eval(_in))
                except:
                    out = exec(_in)
                    if out != None:
                        print(out)
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt as e:
        print("\nExiting...")


if __name__ == '__main__':
	main()       
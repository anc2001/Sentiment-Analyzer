from main import get_rating
from tensorflow.keras.models import load_model

def main():
    try:
        model = load_model('models/model_save.h5')
        while True:
            try:
                _in = input(">> ")
                try:
                    rating = get_rating(model,_in)
                    text = '\n Rating: ' + rating
                    print(text)
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
from cryptography.fernet import Fernet

def load_key():
    return open("encryption_key.key", "rb").read()

def encrypt_file(filename, key):
    fernet = Fernet(key)
    with open(filename, "rb") as file:
        file_data = file.read()
    encrypted_data = fernet.encrypt(file_data)
    with open(filename + ".encrypted", "wb") as file:
        file.write(encrypted_data)

if __name__ == "__main__":
    # Load the encryption key
    key = load_key()
    
    # Encrypt the file
    file_to_encrypt = "locked_data.txt"  # Replace with your file path
    encrypt_file(file_to_encrypt, key)
    
    print(f"File '{file_to_encrypt}' encrypted successfully.")
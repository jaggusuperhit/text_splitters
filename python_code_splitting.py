from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
class BankAccount:
    def __init__(self, account_number, account_name, balance=0):
        self.account_number = account_number
        self.account_name = account_name
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited {amount}. New balance: {self.balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew {amount}. New balance: {self.balance}")
        elif amount <= 0:
            print("Invalid withdrawal amount.")
        else:
            print("Insufficient funds.")

    def display_details(self):
        print(f"Account Number: {self.account_number}")
        print(f"Account Name: {self.account_name}")
        print(f"Balance: {self.balance}")


# Example usage:
account = BankAccount("123456789", "John Doe", 1000)
account.display_details()
account.deposit(500)
account.withdraw(200)
account.display_details()
"""

# Create the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

# Split the text
chunks = splitter.split_text(text)

# Display results
print(len(chunks))  # Number of chunks
print(chunks[1])    # Second chunk

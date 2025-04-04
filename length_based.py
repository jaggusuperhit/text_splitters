# Import the CharacterTextSplitter class
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

# # Define the input text
# text = """
# The quick brown fox jumps over the lazy dog. The sun is shining brightly in the clear blue sky. The birds are singing their sweet melodies, and the flowers are blooming beautifully.

# The gentle breeze is blowing softly, and the trees are swaying gently. The world is full of beauty and wonder. Everything is so peaceful and serene, it's a perfect day to relax and enjoy nature.
# """

# Initialize the CharacterTextSplitter with desired parameters
splitter = CharacterTextSplitter(
    chunk_size=100,  # Split text into chunks of 100 characters
    chunk_overlap=0,  # No overlap between chunks
    separator=''  # No separator between chunks
)

# Split the text into chunks
# result = splitter.split_text(text)
result = splitter.split_documents(docs)

# Print the result
print(result[0])
# print(result)
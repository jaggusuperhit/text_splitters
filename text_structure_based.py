from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
The quick brown fox jumps over the lazy dog. The sun is shining brightly in the clear blue sky. The birds are singing their sweet melodies, and the flowers are blooming beautifully.

The gentle breeze is blowing softly, and the trees are swaying gently. The world is full of beauty and wonder. Everything is so peaceful and serene, it's a perfect day to relax and enjoy nature.

The stars are shining brightly in the night sky, and the moon is glowing with a soft, silvery light. The world is full of magic and mystery, and anything is possible.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)
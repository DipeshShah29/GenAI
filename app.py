import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile

# ----------------------------
# Output Schema
# ----------------------------
class StoryOutput(BaseModel):
    Topic: str = Field(description="Title or topic of the story")
    Story: str = Field(description="Age-appropriate story content")
    Moral: str = Field(description="Moral of the story")

story_parser = PydanticOutputParser(pydantic_object=StoryOutput)

# ----------------------------
# Prompt
# ----------------------------
# 2. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative, professional and award winning author who writes engaging, exciting, suspense-full"
     "and emotionally rich short stories with strong character arcs and a satisfying ending according to the reader's age."
    "You MUST return ONLY valid JSON. "
 "Do NOT include explanations, markdown, or extra text."
    ),
    ("human",
    """
        Write a 400 words story based on the following details:

        Theme / Topic: {topic}
        Reader Age: {age}
        Genre: {genre}
        Character: {name}

        {format_instructions}

        Age Guidelines:
        - Age 3–6: Very simple words, cheerful tone, short sentences
        - Age 7–12: Simple plot, light adventure, positive moral
        - Age 13–17: Deeper emotions, character growth, mild conflict
        - Age 18+: Mature themes, nuanced characters, richer language

        Ensure the content is fully appropriate for the given age.
        """
    )
]).partial(format_instructions=story_parser.get_format_instructions())

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4
)

# ----------------------------
# Chain
# ----------------------------
story_chain = prompt | llm | story_parser

# ----------------------------
# PDF Generator
# ----------------------------
def generate_pdf(result):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=A4)
    text = c.beginText(40, 800)

    text.setFont("Helvetica-Bold", 14)
    text.textLine(result.Topic)
    text.textLine("")

    text.setFont("Helvetica", 11)
    for line in result.Story.split("\n"):
        text.textLine(line)

    text.textLine("\nMoral:")
    text.textLine(result.Moral)

    c.drawText(text)
    c.showPage()
    c.save()
    return temp_file.name

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Story Generator", layout="centered")
st.title("📖 AI Story Generator")

user_topic = st.text_input("Enter a topic, theme, or idea for the story")
genre = st.selectbox(
    "Select a genre",
    ["Adventure", "Educational","Fairy Tale", "Fantasy",   "Fiction","Motivational","Mystery", "Sci-Fi"]
)

language = st.selectbox(
    "Select story language",
    ["English", "Hindi", "Gujarati"]
)

age = st.number_input("Reader's age", min_value=3, max_value=100, step=1)
name = st.text_input("Main character's name")

parent_safe = False if age > 18 else st.checkbox("🧒 Parent-safe content mode", value=True)

if st.button("✨ Generate Story"):
    if not user_topic or not name:
        st.warning("Please fill all required fields.")
    else:
        with st.spinner("Creating your story..."):
            try:
                result = story_chain.invoke({
                    "topic": user_topic,
                    "genre": genre,
                    "language": language,
                    "age": age,
                    "name": name,
                    "parent_safe": parent_safe
                })

                st.subheader("📌 Topic")
                st.write(result.Topic)

                st.subheader("📚 Story")
                st.write(result.Story)

                if genre == "Motivational" or genre == "Educational":
                    st.subheader("🌱 Moral")
                    st.write(result.Moral)

                # PDF Download
                pdf_path = generate_pdf(result)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📄 Download as PDF",
                        data=f,
                        file_name="story.pdf",
                        mime="application/pdf"
                    )

            except Exception as e:
                st.error("Story generation failed.")
                st.exception(e)

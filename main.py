import streamlit as st
from groq import Groq
import json
import os
from io import BytesIO
from md2pdf.core import md2pdf
from dotenv import load_dotenv
from download import download_video_audio, delete_download
from pydub import AudioSegment

load_dotenv()

# idk where it is in the code but for file upload, if we're given a video then we can just split it using python libraries and get the audio.

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
FILE_TOO_LARGE_MESSAGE = "The audio file is too large for the current size and rate limits using Whisper. If you used a YouTube link, please try a shorter video clip. If you uploaded an audio file, try trimming or compressing the audio to under 25 MB."

global_variable = None
audio_file_path = None

if 'api_key' not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY

if 'groq' not in st.session_state:
    if GROQ_API_KEY:
        st.session_state.groq = Groq()

st.set_page_config(
    page_title="BlogWizard",
    page_icon="🧙‍♂️",
)
      
class GenerationStatistics:
    def __init__(self, input_time=0,output_time=0,input_tokens=0,output_tokens=0,total_time=0,model_name="llama3-8b-8192"):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time # Sum of queue, prompt (input), and completion (output) times
        self.model_name = model_name

    def get_input_speed(self):
        """ 
        Tokens per second calculation for input
        """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0
    
    def get_output_speed(self):
        """ 
        Tokens per second calculation for output
        """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0
    
    def add(self, other):
        """
        Add statistics from another GenerationStatistics object to this one.
        """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")
        
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n\n"
                f"| Metric          | Input          | Output          | Total          |\n"
                f"|-----------------|----------------|-----------------|----------------|\n"
                f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
                f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {self.input_tokens + self.output_tokens}            |\n"
                f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |")

class NoteSection:
    def __init__(self, structure, transcript):
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}

        st.markdown("## Raw transcript:")
        st.markdown(transcript)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def return_existing_contents(self, level=1) -> str:
        existing_content = ""
        for title, content in self.structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                existing_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                existing_content += self.get_markdown_content(content, level + 1)
        return existing_content

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        
        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index
    
    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure
        
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content
    
    def get_markdown_arabic(self, structure=None, level=1):
        """
        Returns the dictionary contents of the structure.
        """
        if structure is None:
            structure = self.structure
        
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += translate_to_arabic(f"{'#' * level} {title}\n{self.contents[title]}.\n\n")
            if isinstance(content, dict):
                markdown_content += translate_to_arabic(self.get_markdown_content(content, level + 1))
        return markdown_content


def create_markdown_file(content: str) -> BytesIO:
    """
    Create a Markdown file from the provided content.
    """
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

def create_pdf_file(content: str):
    """
    Create a PDF file from the provided content.
    """
    pdf_buffer = BytesIO()
    md2pdf(pdf_buffer, md_content=content)
    pdf_buffer.seek(0)
    return pdf_buffer

def transcribe_audio(audio_file):
    """
    Transcribes audio using Groq's Whisper API.
    """
    transcription = st.session_state.groq.audio.transcriptions.create(
      file=audio_file,
      model="whisper-large-v3",
      prompt="If Groq is mentioned it is spelled Groq",
      response_format="json",
      language="en",
      temperature=0.0 
    )

    results = transcription.text
    return results

def generate_notes_structure(transcript: str, blog_style, model: str = "llama3-70b-8192"):
    """
    Returns notes structure content as well as total tokens and total time for generation.
    """

    shot_example = """
    "Introduction": "Brief overview of the topic. Why it's relevant and important",
    "Key Topic Discussions [1-3]": "Talk about the key moments of the topic",
    "Analysis and Insights": "Highlight insights and statistics. May include past and present comparison",
    "Takeaways": "Share advice that may be relevant to the readers",
    "Conclusion": "May include recap of key points, implications for the future, call to action."
    }"""
    if blog_style == "Customer Case Study":
        shot_example = """
        Customer company description
        Challenge
        -List one to three main challenges that a customer or an end user faces
        -These challenges should clearly express why there is a need for solution
        Solution
        -Explain how the featured customer solves the above stated challenge
        -In this description, include unique advantages and specific ROI that the customer offers to its end users
        -Explain how Groq enables this customer to deliver this solution better than anyone else - this should usually include something about our value prop around speed, scalability, performance, or ROI
        Key Features
        Opportunity
        -Explain the ways this solution can transform an end user's experience, disrupt an industry, or change the course of the world
        -Explain how the solution can be applied to various industries and use cases"""
    elif blog_style == "Launch of new Product":
        shot_example = """
        Introduction
        -Name of model is now available on GroqCloud
        -How to access the model
        -A video or image showcasing the model running
        -Quote from senior level executive, internal or external
        Advantages of model
        -Speed
        -Quality
        -Performance
        -Price
        -Third party benchmarks if available
        Background on the model
        -How was it built?
        -Who does it serve?
        -What use cases can it help with most?
        -Why does it matter that the model is running on Groq
        -Name of model running on GroqCloud means (speed, accessibility, performance, or some other value prop) for developers -and enterprises that is otherwise unavailable in the market
        CTA
        -Start building with Name of model today
        -Call out any tools or features that make the model more enticing (tool use, higher rate limits, etc)
"""
    
    completion = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Write in JSON format:\n\n{\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\"}"
            },
            {
                "role": "user",
                "content": f"### Transcript {transcript}\n\n### Example\n\n{shot_example}### Instructions\n\nCreate a structure for a comprehensive blog article on the above transcribed audio. Section titles and content descriptions must be comprehensive. Quality over quantity."
            }
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name=model)

    return statistics_to_return, completion.choices[0].message.content

def generate_section(blog_length, transcript: str, existing_notes: str, section: str, model: str = "llama3-8b-8192"):
    stream = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert blog writer. Generate body content in third-person for the section provided based on the transcript. Do *not* repeat any content from previous sections. No need to preface with any titles or pleasantries, just provide the paragraphs. Max word count of {blog_length} words."
            },
            {
                "role": "user",
                "content": f"### Transcript\n\n{transcript}\n\n### Existing Notes\n\n{existing_notes}\n\n### Instructions\n\nGenerate short blog-like paragraphs only for this section based on the transcript: \n\n{section}"
            }
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics_to_return = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name=model)
            yield statistics_to_return

# Initialize
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

if 'button_text' not in st.session_state:
    st.session_state.button_text = "Generate Blog"

if 'statistics_text' not in st.session_state:
    st.session_state.statistics_text = ""

if 'buttons_misc_disabled' not in st.session_state:
    st.session_state.buttons_misc_disabled = True

if 'notes' not in st.session_state:
    st.session_state.notes = None
# if 'notes_structure_json' not in st.session_state:
#     st.session_state.notes_structure_json = {}

st.write("""
# BlogWizard: Create structured blog from audio 🗒️⚡
""")

def enable_buttons_misc():
    st.session_state.buttons_misc_disabled = False

def disable():
    st.session_state.button_disabled = True
    # and also enable the miscs buttons
    st.session_state.buttons_misc_disabled = False

def enable():
    st.session_state.button_disabled = False

def empty_st():
    st.empty()

def translate(text, selected_lang):
    chat_completion = st.session_state.groq.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"Translate this text into {selected_lang}. Use markdown."
        },
        {
            "role": "user",
            "content": text,
        }
    ],
    model="llama-3.3-70b-versatile"
    )
    print(f"translated notes in {selected_lang}: ", chat_completion.choices[0].message.content)
    
    return chat_completion.choices[0].message.content

def translate_to_arabic(markdown_content):
           
    chat_completion = st.session_state.groq.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Translate the entire text into Arabic"
        },
        {
            "role": "user",
            "content": markdown_content,
        }
    ],
    model="allam-2-7b",
    )
    print("translated notes: ", chat_completion.choices[0].message.content)
    
    return chat_completion.choices[0].message.content


image_file = "assets/groqlabs.svg"
try:
    with st.sidebar:

        if image_file:
            st.image(image_file, width=200)
        
        st.write(f"# 🧙‍♂️ BlogWizard \n## Generate blog from audio in seconds using Groq, Whisper, and Llama3")
        st.markdown(f"[Github Repository](https://github.com/cho-groq/BlogWizard)\n\n")

        STYLES = [
            "Default",
            "Customer Case Study",
            "Launch of new Product"
        ]

        st.title("Blog options")

        # Create a dropdown selector
        blog_style = st.selectbox("Choose a template style:", options=STYLES)

        BLOG_WORD_COUNT = {
            "Up to 800 words":200,
            "Up to 1400 words":300,
            "Up to 2500 words":500,
        }

        # Create a dropdown selector
        blog_length = st.selectbox("Choose a word count:", options=BLOG_WORD_COUNT.keys())

        st.info("Audio files and YouTube videos over 19 minutes will be summarized only up to the first 19 minutes. Videos longer than 3 hours are not allowed")

        audio_files = {
            "Groq AI Weekly Updates": {
                "file_path": "assets/audio/groq_ama_trimmed_20min.m4a",
                "youtube_link": "https://www.youtube.com/watch?v=A3IRU6aoLYA"
            },
            "Highlights of 2025 LIV Golf Riyadh Round 1": {
                "file_path": "assets/audio/transformers_explained.m4a",
                "youtube_link": "https://www.youtube.com/watch?v=SZorAJ4I-sA"
            },
            "Joaquin Niemann LIV Golf Adelaide Postgame Winner Interview": {
                "file_path": "assets/audio/essence_calculus.m4a",
                "youtube_link": "https://www.youtube.com/watch?v=xIVKjjKQgl4"
            }
        }

        st.write(f"---")

        st.write(f"# Sample Audio Files")

        for audio_name, audio_info in audio_files.items():

            st.write(f"### {audio_name}")
            
            # Read audio file as binary
            with open(audio_info['file_path'], 'rb') as audio_file:
                audio_bytes = audio_file.read()

            # Create download button
            # st.download_button(
            #     label=f"Download audio",
            #     data=audio_bytes,
            #     file_name=audio_info['file_path'],
            #     mime='audio/m4a'
            # )
            
            st.markdown(f"[Youtube Link]({audio_info['youtube_link']})")
            st.write(f"\n\n")
        
        st.write(f"---")

        st.write("# Customization Settings\n🧪 These settings are experimental.\n")
        st.write(f"By default, BlogWizard uses Llama3.3-70b for generating the blog outline and Llama3-8b for the content. This balances quality with speed and rate limit usage. You can customize these selections below.")
        outline_model_options = ["llama-3.3-70b-versatile", "llama3-70b-8192", "deepseek-r1-distill-qwen-32b", "mixtral-8x7b-32768", "gemma-9b-it"]
        outline_selected_model = st.selectbox("Outline generation:", outline_model_options)
        content_model_options = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
        content_selected_model = st.selectbox("Content generation:", content_model_options)

        
        # Add note about rate limits
        st.info("Important: Different models have different token and rate limits which may cause runtime errors.")

        LANGUAGES = {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "hi": "Hindi",
            "nl": "Dutch",
            "sv": "Swedish",
            "fi": "Finnish",
            "da": "Danish",
            "no": "Norwegian",
            "pl": "Polish",
            "tr": "Turkish",
            "he": "Hebrew",
        }

        st.title("Language Translate")

        # Create a dropdown selector
        selected_lang = st.selectbox("Choose a language to translate to:", options=LANGUAGES.values())

        @st.dialog(f"{selected_lang} translation", width="large")
        def language(item):
            st.markdown(item)
        
       
        # Get the abbreviation code from the selected language. but not needed for text
        # selected_code = next(code for code, name in LANGUAGES.items() if name == selected_lang)


        if selected_lang:
            if st.button("Translate into language", disabled=st.session_state.buttons_misc_disabled):
                translation = translate(st.session_state.notes.get_markdown_content(), selected_lang)
                language(translation)
                
       
        st.title("Translate into Arabic:")

        @st.dialog("Arabic Translation", width="large")
        def arabic(item):
            st.markdown(
            f'<div dir="rtl" style="text-align: right;">{item}</div>',
            unsafe_allow_html=True
           )
        
        if "arabic" not in st.session_state:
            if st.button("Translate into Arabic", disabled=st.session_state.buttons_misc_disabled):
                arabic_translation = st.session_state.notes.get_markdown_arabic()
                print(arabic_translation)
                arabic(arabic_translation)

        def linkedin_post(text, selected_lang, social_media):
            chat_completion = st.session_state.groq.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"Create a social post in the style of {social_media}. Use markdown and emojis.{' Make it less than 280 characters.' if social_media == 'X' else ''}",

                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="llama-3.3-70b-versatile",
            )

            temp = chat_completion.choices[0].message.content
            chat_completion2 = st.session_state.groq.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"Translate this markdown text into {selected_lang}:",
                },
                {
                    "role": "user",
                    "content": temp,
                }
            ],
            model="llama-3.3-70b-versatile",
            )
            
            return chat_completion2.choices[0].message.content
        
        @st.dialog("Social Media Post", width="large")
        def vote(item):
            # Path = f'''{item}'''
            st.markdown(item)
        
        
        social_media_options = ["LinkedIn", "Facebook", "X", "Instagram", "Reddit"]
        st.title("Turn into a Social Media post")
        social_media = st.selectbox("Choose a social media platform:", social_media_options)
        st.write("Also uses the langugage above to translate.")
        
        if "vote" not in st.session_state:
            if st.button("Create Social Media post", disabled=st.session_state.buttons_misc_disabled):
                linkedin_post_text = linkedin_post(st.session_state.notes.get_markdown_content(), selected_lang, social_media)
                vote(linkedin_post_text)
                
        st.markdown("""
        - [Groq Terms of Use](https://groq.com/terms-of-use/)
        - [Groq Privacy Policy (PDF)](https://groq.com/wp-content/uploads/2024/05/Groq-Privacy-Policy_Final_30MAY2024.pdf)
        """)
        
    if st.button('End Generation and Download Blog'):
        if "notes" in st.session_state:

            # Create markdown file
            markdown_file = create_markdown_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download Text',
                data=markdown_file,
                file_name='generated_notes.txt',
                mime='text/plain'
            )

            # Create pdf file (styled)
            pdf_file = create_pdf_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download PDF',
                data=pdf_file,
                file_name='generated_notes.pdf',
                mime='application/pdf'
            )
            st.session_state.button_disabled = False
        else:
            raise ValueError("Please generate content first before downloading the blog.")

    input_method = st.radio("Choose input method:", ["Upload audio file", "YouTube link"])
    audio_file = None
    youtube_link = None
    groq_input_key = None
    with st.form("groqform"):
        if not GROQ_API_KEY:
            groq_input_key = st.text_input("Enter your Groq API Key (gsk_yA...):", "", type="password", autocomplete="off")
        
        # Add radio button to choose between file upload and YouTube link
        
        if input_method == "Upload audio file":
            audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"]) # TODO: Add a max size
        else:
            youtube_link = st.text_input("Enter YouTube link:", "")

        # Generate button
        submitted = st.form_submit_button(st.session_state.button_text, on_click=disable, disabled=st.session_state.button_disabled)

        #processing status
        status_text = st.empty()
        def display_status(text):
            status_text.write(text)

        def clear_status():
            status_text.empty()

        download_status_text = st.empty()
        def display_download_status(text:str):
            download_status_text.write(text)    

        def clear_download_status():
            download_status_text.empty()
        
        # Statistics
        placeholder = st.empty()
        def display_statistics():
            with placeholder.container():
                if st.session_state.statistics_text:
                    if "Transcribing audio in background" not in st.session_state.statistics_text:
                        st.markdown(st.session_state.statistics_text + "\n\n---\n")  # Format with line if showing statistics
                    else:
                        st.markdown(st.session_state.statistics_text)
                else:
                    placeholder.empty()

        # this displays the notes on the second go around when the user clicks a button on the side of the page
        if 'notes' in st.session_state and st.session_state.notes is not None:
            st.markdown(st.session_state.notes.get_markdown_content())

        if submitted:
            if input_method == "Upload audio file" and audio_file is None:
                st.error("Please upload an audio file")
            elif input_method == "YouTube link" and not youtube_link:
                st.error("Please enter a YouTube link")
            else:
                st.session_state.button_disabled = True
                # Show temporary message before transcription is generated and statistics show
            
            audio_file_path = None

            if input_method == "YouTube link":
                display_status("Downloading audio from YouTube link ....")
                audio_file_path = download_video_audio(youtube_link, display_download_status)
                if audio_file_path is None:
                    st.error("Failed to download audio from YouTube link. Please try again.")
                    enable()
                    clear_status()
                else:
                    # Read the downloaded file and create a file-like objec
                    display_status("Processing Youtube audio ....")

                    # Check size first to ensure will work with Whisper
                    if os.path.getsize(audio_file_path) > MAX_FILE_SIZE:
                        # use pydub to get the first 15 minutes of the audio file
                        print(FILE_TOO_LARGE_MESSAGE)
                        audio = AudioSegment.from_file(audio_file_path)
                            
                        # Extract the first 19 minutes
                        fifteen_minutes_in_ms = 19 * 60 * 1000  # pydub works in milliseconds
                        trimmed_audio = audio[:fifteen_minutes_in_ms]
                        
                        # Export directly to the original file path, overwriting it
                        trimmed_audio.export(audio_file_path, format="mp3")
                        
                    # Now read the file (either original or trimmed) into memory
                    with open(audio_file_path, 'rb') as f:
                        file_contents = f.read()
                    audio_file = BytesIO(file_contents)

                    audio_file.name = os.path.basename(audio_file_path)  # Set the file name
                    delete_download(audio_file_path)
                clear_download_status()

            if not GROQ_API_KEY:
                st.session_state.groq = Groq(api_key=groq_input_key)

            display_status("Transcribing audio in background....")
            transcription_text = transcribe_audio(audio_file)

            display_statistics()
            

            display_status("Generating blog structure....")
            large_model_generation_statistics, notes_structure = generate_notes_structure(transcription_text, blog_style, model=str(outline_selected_model))
            # print("Structure: ",notes_structure)

            display_status("Generating blog ...")
            total_generation_statistics = GenerationStatistics(model_name=str(content_selected_model))
            clear_status()


            try:
                notes_structure_json = json.loads(notes_structure)
                st.session_state.notes_structure_json = notes_structure_json
                # print(notes_structure_json)
                notes = NoteSection(structure=notes_structure_json,transcript=transcription_text)

                st.session_state.notes = notes

                st.session_state.notes.display_structure()
                print( st.session_state.notes.display_structure())

                # will this save the notes
                # st.session_state.markdown = st.session_state.notes.get_markdown_content()
                # print("this is the markdown: "+st.session_state.markdown)

                # st.write(st.session_state.markdown)
    
                # st.markdown(st.session_state.notes.get_markdown_content())

                # st.markdown(st.session_state.markdown)

                def stream_section_content(sections):
                    for title, content in sections.items():
                        if isinstance(content, str):
                            content_stream = generate_section(blog_length, transcript=transcription_text, existing_notes=notes.return_existing_contents(), section=(title + ": " + content),model=str(content_selected_model))
                            for chunk in content_stream:
                                # Check if GenerationStatistics data is returned instead of str tokens
                                chunk_data = chunk
                                if type(chunk_data) == GenerationStatistics:
                                    total_generation_statistics.add(chunk_data)
                                    
                                    st.session_state.statistics_text = str(total_generation_statistics)
                                    display_statistics()
                                elif chunk is not None:
                                    st.session_state.notes.update_content(title, chunk)
                        elif isinstance(content, dict):
                            stream_section_content(content)

                stream_section_content(notes_structure_json)
                # st.write(st.session_state.notes)
                # st.write("NONONONONON")
                # st.markdown(st.session_state.notes)

            except json.JSONDecodeError:
                st.error("Failed to decode the blog structure. Please try again.")

            enable()
            

except Exception as e:
    st.session_state.button_disabled = False

    if hasattr(e, 'status_code') and e.status_code == 413:
        # In the future, this limitation will be fixed as BlogWizard will automatically split the audio file and transcribe each part.
        st.error(FILE_TOO_LARGE_MESSAGE)
    else:
        st.error(e)

    if st.button("Clear"):
        st.rerun()
    
    # Remove audio after exception to prevent data storage leak
    if audio_file_path is not None:
        delete_download(audio_file_path)

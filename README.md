<h2 align="center">
 <br>
 <img src="https://i.imgur.com/scoiUgD.png" alt="Generate Organizes Notes with BlogWizard" width="150">
 <br>
 <br>
 BlogWizard: Generate blog articles from video or audio<br>using Groq, Whisper, and Llama/Kimi/Deepseek
 <br>
</h2>


<p align="center">
 <a href="https://github.com/cho-groq/BlogWizard/stargazers"><img src="https://img.shields.io/github/stars/cho-groq/BlogWizard"></a>
 <a href="https://github.com/cho-groq/BlogWizard/blob/main/LICENSE.md">
 <img src="https://img.shields.io/badge/License-MIT-green.svg">
 </a>
</p>

<p align="center">
 <a href="#Overview">Overview</a> •
 <a href="#Features">Features</a> •
 <a href="#Quickstart">Quickstart</a> •
 <a href="#Contributing">Contributing</a>
</p>

<br>

[Demo of BlogWizard](https://github.com/user-attachments/assets/0893d952-bb3b-4f94-b79d-cedb6183024b)
> Demo of BlogWizard fast transcription of audio and generation of structured blog.


## Overview

BlogWizard is a streamlit app that scaffolds the creation of blogs by iteratively structuring and generating blogs from transcribed audio lectures using Groq's Whisper API.


### Features

- 🎧 Generate a structured blog using transcribed audio by Whisper-large and text by Llama, Kimi, or Deepseek
- ⚡ Lightning fast speed transcribing audio and generating text using Groq
- 📖 Scaffolded prompting strategically switches between Content and Outline models to balance speed and quality
- 🖊️ Markdown styling creates aesthetic blog posts on the streamlit app that can include tables and code 
- 📂 Allows user to download a text or PDF file with the entire blog contents

## Quickstart

> [!IMPORTANT]
> To use BlogWizard, you can use a hosted version at [https://blogwizard.groqlabs.com/](https://blogwizard.groqlabs.com/).
> Alternatively, you can run BlogWizard locally with Streamlit using the quickstart instructions.


### Run locally:

Alternative, you can run BlogWizard locally with streamlit.

#### Step 1
First, you can set your Groq API key in the environment variables:

~~~
export GROQ_API_KEY="gsk_yA..."
~~~

This is an optional step that allows you to skip setting the Groq API key later in the streamlit app.

#### Step 2
Next, you can set up a virtual environment and install the dependencies.

~~~
python3 -m venv venv
~~~

~~~
source venv/bin/activate
~~~

~~~
pip3 install -r requirements.txt
~~~


#### Step 3
Finally, you can run the streamlit app.

~~~
python3 -m streamlit run main.py
~~~

## Details


### Technologies

- Streamlit
- Llama, Kimi, and Deepseek on Groq Cloud
- Whisper-large on Groq Cloud
- PyDub to take the first 
- It is recommented to use python3.12

### Limitations

Audio files greater than 25mb or YouTube videos longer than 19 minutes will be trimmed down to those thresholds amounts listed and then summarized.

BlogWizard may generate inaccurate information or placeholder content. It should be used to generate notes for entertainment purposes only.

## Contributing

Improvements through PRs are welcome!

Adapted from ScribeWizard by [Ben Klieger](https://github.com/bklieger-groq)

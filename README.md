# RecipeFindr | Your Real-time Kitchen Agent
You're looking at "RecipeFindr"! This is an agentic model that takes a user's image of their ingredients and creates a couple of recipes for them to follow. A group of two of my peers from MITES and I built this model using Agno, Groq, and Spoonacular API.

We used Agno as our framework as it allowed our agents to work together in a seemless process. We used Groq as it allowed us to access API calls for free. We used Spoonacular API as it provided a database with access to thousands of recipes and have functions to search for ingredients.


# How to Use
1. Enter your image path. Due to Agno's Playground only accepting textual input, you must submit an online image. To do this, copy its image address and paste that into the user input box.
       ex. "https://mites.mit.edu/files/2025/05/MitesLogo.png" (must have the quotes)
3. Once the model returned the ingredients detected, you can now input different ingredients that were missed or mistakenly identified. Follow the instructions given by the model to change your ingredients list.
4. Once you've finished your list, tell the model that you've completed your list of ingredients and now you can generate a couple of recipes to optimize ingredients in your fridge!


# Intructions to Setup
Before you use this, you will need to create your own API Key for your agentic models (Groq, OpenAI, etc.) and store this in an .env file. After that, the code should be able to work.

To use this, you need to set up your Agno Playground. You can do this by inputting "ag setup" in your terminal. Afterwards, you can access the workflow in Agno's PLayground. Once you have that set up, you can start interacting with our agents!

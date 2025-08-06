# Import required libraries for AI workflow, computer vision, and API integration
from textwrap import dedent
import os
import ast
import json
import requests
from dotenv import load_dotenv
from ultralytics import YOLO
from agno.workflow import Workflow, RunResponse
from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground


# Load environment variables for API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
spoon_api_key = os.getenv("SPOONACULAR_API_KEY")


# Global memory to store workflow state and data between interactions
WORKFLOW_MEMORY = {}


# Example image paths for testing
# img_path = "C:\\Users\\IT2\\Downloads\\Fridge #3.jfif"
#img_path = "https://i.postimg.cc/Hny1qW0S/Fridge-3.jpg"


# Computer vision function to detect food items in images using YOLO model
def get_ingredients(img_path: str) -> list[str]:
  import cv2, numpy as np, requests
  from ultralytics import YOLO


  model = YOLO("yolov8m.pt")
  if img_path.startswith("http"):
      response = requests.get(img_path)
      image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
      img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
  else:
      img = cv2.imread(img_path)


  if img is None:
      raise ValueError("Failed to load image.")


  results = model(img)[0]
  labels = results.names
  boxes = results.boxes


  detected_items = [labels[int(b.cls)] for b in boxes]
  return list(set(detected_items))


# API function to fetch recipes from Spoonacular based on available ingredients
def get_recipes_from_ingredients(ingredients: list[str], number: int = 3):
  url = "https://api.spoonacular.com/recipes/findByIngredients"
  params = {
      "ingredients": ",".join(ingredients),
      "number": number,
      "ranking": 2,
      "apiKey": spoon_api_key,
  }
  response = requests.get(url, params=params)
  if response.status_code == 200:
      return response.json()
  else:
      return {"error": f"API call failed with status code {response.status_code}: {response.text}"}


# Helper function to parse ingredient lists from AI agent responses
def parse_ingredient_list(raw_response: str) -> list[str]:
  try:
      import re
      list_match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
      if list_match:
          list_str = list_match.group()
          ingredient_list = ast.literal_eval(list_str)
          if isinstance(ingredient_list, list):
              return [str(item).strip() for item in ingredient_list if item]


      lines = [line.strip().strip("',\".-*")
               for line in raw_response.split('\n')
               if line.strip() and not line.strip().startswith('#')]
      return [item for item in lines if item and len(item) > 1]
  except:
      return [raw_response.strip()]


# Utility functions to determine user intent from input text
def is_image_path(text: str) -> bool:
  return any(ext in text.lower() for ext in ['.jpg', '.jpeg', '.png', '.jfif', 'http']) or 'fridge' in text.lower()


def is_validation_input(text: str) -> bool:
  validation_keywords = ['looks good', 'remove', 'add', 'delete', 'change', 'keep', 'ok', 'good', 'fine', 'perfect']
  return any(keyword in text.lower() for keyword in validation_keywords)


def is_recipe_request(text: str) -> bool:
  recipe_keywords = ['recipe', 'generate', 'cook', 'make', 'find', 'create', 'show', 'get']
  return any(keyword in text.lower() for keyword in recipe_keywords)


# Main workflow class that orchestrates the entire recipe generation process
class SmartRecipeGenerator(Workflow):
  name = "🍳 Smart Recipe Generator"
  description = "Intelligent conversational recipe generator with memory"


  # AI agent for analyzing images and detecting ingredients using computer vision
  vision_agent = Agent(
      name="Vision Analyzer Agent",
      role="Detect ingredients from an image",
      model=Groq(id="llama3-70b-8192", api_key=groq_api_key),
      tools=[get_ingredients],
      instructions=dedent("""
          Call get_ingredients(img_path) with the provided image path.
          Return the result as a clean Python list format.
          Example response: ["apple", "banana", "carrot", "milk"]


          IMPORTANT: Your response must be ONLY the Python list, nothing else.
          Do not add explanatory text, just the list.
      """),
      show_tool_calls=True
  )


  # AI agent for validating and modifying ingredient lists based on user feedback
  validator_agent = Agent(
      name="Ingredient Validator Agent",
      description='Validates and modifies ingredient lists based on user input',
      model=Groq(id='llama3-70b-8192', api_key=groq_api_key),
      instructions=dedent("""
          You will receive:
          1. An original ingredient list
          2. User modifications/requests


          Process the user's request and return ONLY a clean Python list.
          Examples:
          - If user says "remove banana, add cheese": modify the list accordingly
          - If user says "looks good": return the original list unchanged
          - If user says "add rice and eggs": add those items


          Your response must be ONLY a Python list like: ["apple", "rice", "eggs"]
          No explanations, no formatting, just the list.
      """)
  )


  # AI agent for generating formatted recipes using the Spoonacular API
  recipe_agent = Agent(
      name="Recipe Generator Agent",
      role="Generate recipes from ingredients",
      model=Groq(id="llama3-70b-8192", api_key=groq_api_key),
      tools=[get_recipes_from_ingredients],
      instructions=dedent("""
          You will receive a list of ingredients.


          Process the ingredients:
          1. Remove any non-food items
          2. Add plural versions where appropriate (e.g., if "orange" add "oranges")
          3. Call get_recipes_from_ingredients with the processed list
          4. Format the response nicely with recipe titles, ingredients, and step-by-step instructions


          Format each recipe as:
          # Recipe Title


          ## Ingredients Used
          • Ingredient 1 - amount
          • Ingredient 2 - amount


          ## Instructions
          1. Step one
          2. Step two
          3. Step three


          If no recipes are found, suggest alternatives or modifications.
      """),
      show_tool_calls=True,
      markdown=True
  )


  # Main workflow logic that handles state management and user interactions
  def run(self, user_input: str):
      current_state = WORKFLOW_MEMORY.get('state', 'waiting_for_image')
      stored_ingredients = WORKFLOW_MEMORY.get('ingredients', [])


      # Handle image processing and ingredient detection
      if current_state == 'waiting_for_image' or is_image_path(user_input):
          vision_resp = self.vision_agent.run(user_input)
          if not vision_resp or not hasattr(vision_resp, "content"):
              yield RunResponse(content="❌ Error: Could not process the image. Please check the image path.")
              return


          raw_ingredients = vision_resp.content.strip()
          ingredient_list = parse_ingredient_list(raw_ingredients)


          WORKFLOW_MEMORY['ingredients'] = ingredient_list
          WORKFLOW_MEMORY['state'] = 'waiting_for_validation'


          ingredients_display = "\n".join([f"• {item}" for item in ingredient_list])


          response_content = f"""## 🥕 Detected Ingredients:


{ingredients_display}


**Detected {len(ingredient_list)} ingredients from your image.**


**Examples:**
- `looks good` (keep all ingredients)
- `remove banana, add cheese`
- `add rice and eggs, remove carrot`


**What would you like to do with this ingredient list?**"""


          yield RunResponse(content=response_content)
          return


      # Handle ingredient list validation and modifications
      elif current_state == 'waiting_for_validation' and stored_ingredients:
          validation_prompt = f"""Original ingredients: {stored_ingredients}
User request: "{user_input}"
Update the ingredient list based on the user's request."""


          validator_resp = self.validator_agent.run(validation_prompt)
          if not validator_resp or not hasattr(validator_resp, "content"):
              yield RunResponse(content="❌ Error: Could not validate ingredients.")
              return


          validated_raw = validator_resp.content.strip()
          validated_list = parse_ingredient_list(validated_raw)


          WORKFLOW_MEMORY['validated_ingredients'] = validated_list
          WORKFLOW_MEMORY['state'] = 'waiting_for_confirmation'


          final_display = "\n".join([f"• {item}" for item in validated_list])


          response_content = f"""## ✅ Updated Ingredient List:


{final_display}


**Is this your final ingredient list?**


**Options:**
- `yes` or `looks good` (proceed to recipes)
- `make more changes` (modify again)
- `add/remove more items`


**What would you like to do?**"""


          yield RunResponse(content=response_content)
          return


      # Handle final confirmation before recipe generation
      elif current_state == 'waiting_for_confirmation' and WORKFLOW_MEMORY.get('validated_ingredients'):
          if any(word in user_input.lower() for word in ['yes', 'good', 'looks good', 'perfect', 'ok', 'correct', 'final', 'proceed']):
              WORKFLOW_MEMORY['state'] = 'waiting_for_recipe_request'
              final_ingredients = WORKFLOW_MEMORY.get('validated_ingredients')
              final_display = "\n".join([f"• {item}" for item in final_ingredients])


              response_content = f"""## ✅ Final Ingredient List Confirmed:


{final_display}


**Ready to generate recipes with {len(final_ingredients)} ingredients!**


**Type:**
- `generate recipes`
- `find recipes`
- `what can I cook?`


**Ready for your recipes?**"""


              yield RunResponse(content=response_content)
              return
          else:
              WORKFLOW_MEMORY['ingredients'] = WORKFLOW_MEMORY.get('validated_ingredients')
              WORKFLOW_MEMORY['state'] = 'waiting_for_validation'


              current_ingredients = WORKFLOW_MEMORY.get('validated_ingredients')
              ingredients_display = "\n".join([f"• {item}" for item in current_ingredients])


              response_content = f"""## 🔄 Current Ingredients:


{ingredients_display}


**Make your changes:**
- `remove banana, add cheese`
- `add rice and eggs`
- `remove carrot`


**What changes would you like to make?**"""


              yield RunResponse(content=response_content)
              return


      # Handle recipe generation and display final results
      elif (current_state == 'waiting_for_recipe_request' and WORKFLOW_MEMORY.get('validated_ingredients')) or is_recipe_request(user_input):
          final_ingredients = WORKFLOW_MEMORY.get('validated_ingredients', stored_ingredients)


          if not final_ingredients:
              yield RunResponse(content="❌ Error: No ingredients found. Please start over with an image path.")
              return


          recipe_resp = self.recipe_agent.run(f"Create recipes using these ingredients: {final_ingredients}")
          if not recipe_resp or not hasattr(recipe_resp, "content"):
              yield RunResponse(content="❌ Error: Could not generate recipes.")
              return


          WORKFLOW_MEMORY.clear()
          WORKFLOW_MEMORY['state'] = 'waiting_for_image'


          response_content = f"""# 🍽️ Recipe Recommendations


{recipe_resp.content}


**To generate recipes for a new image, paste another image path in the text box above!**"""


          yield RunResponse(content=response_content)
          return


      # Handle fallback responses for unclear user input
      else:
          if current_state == 'waiting_for_validation':
              response_content = f"""**Current ingredients:** {', '.join(stored_ingredients)}


**Examples:**
- `looks good` (keep all ingredients)
- `remove banana, add cheese`
- `add rice and eggs, remove carrot`


**What would you like to do?**"""


          elif current_state == 'waiting_for_confirmation':
              validated = WORKFLOW_MEMORY.get('validated_ingredients', [])
              response_content = f"""**Your ingredients:** {', '.join(validated)}


**Is this your final list?**
- `yes` or `looks good` (proceed to recipes)
- `make more changes` (modify again)"""


          elif current_state == 'waiting_for_recipe_request':
              validated = WORKFLOW_MEMORY.get('validated_ingredients', [])
              response_content = f"""**Your ingredients:** {', '.join(validated)}


**Type:**
- `generate recipes`
- `find recipes`
- `what can I cook?`"""


          else:
              response_content = """## 🍳 Smart Recipe Generator


**Start by pasting your fridge image path above! 📸**"""


          yield RunResponse(content=response_content)


# Initialize the workflow instance
smart_recipe_generator = SmartRecipeGenerator(workflow_id='smart_recipe')


# Create the web application using Playground framework
app = Playground(workflows=[smart_recipe_generator]).get_app()


# Run the application server when script is executed directly
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(
      "workflow6:app",
      host="localhost",
      port=9999,
      reload=True
  )


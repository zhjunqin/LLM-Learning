# Large Multimodal Model Prompting with Gemini

https://github.com/https-deeplearning-ai/sc-gc-c4-gemini-public/tree/main

Repo 地址：https://github.com/https-deeplearning-ai/sc-gc-c4-gemini-public/tree/main

## Lesson-2

```
prompt_3 = "Please describe what is in this image?"
file_path = "assets/Andrew_power_tools.png"
output = """
The image shows a man holding a hammer in one hand and a power drill in the other hand. He is smiling and looking at the camera. He is wearing a blue shirt and he has short black hair. The background is a grey wall.
"""


prompt_1 = """Write what is happening in the following image
from a unique perspective and do not mention names"""
file_path = "assets/panda.png"
output = "A man is standing in front of a screen with a stuffed animal in his arms. He is smiling and looking at the camera. There is a slide on the screen that says "Grouping Customers". The man is wearing a blue shirt and black pants. He has short black hair and brown eyes. He is standing in an office space. There is a window in the background."



video_file = "https://storage.googleapis.com/dlai-sc-gemini-bucket/pixel8.mp4"
prompt = """
Answer the following questions using the video only:
 - What is the main person's profession?
 - What are the main features of the phone highlighted?
 - Which city was this recorded in?
"""

output = """
- The main person's profession is a photographer.
- The main features of the phone highlighted are the camera and the night mode.
- This was recorded in Tokyo, Japan.
"""


```

## Lession-3

### Be Clear and Concise
### Assign a Role
```
Role: You are an AI that does image understanding.

Gemini, this dog having a blast playing fetch, caught my eye.
Please provide The breed of dog in the image, and a description of the setting where the dog is
playing.

Answer:
```
### Structure Prompts
```
[image]

Role: You are an AI that does image understanding.

Gemini, this dog having a blast playing fetch, caught my eye.
Please provide The breed of dog in the image,
and a description of the setting where the dog is playing.

Answer:
```
A well-structured prompt does three key things:
1. Organizes Information
2. Guides Interpretation
3. Encourages Desired Output

- Role: Clearly define the role you want the model to take on. This helps it understand its
purpose and adjust its responses accordingly.
- Objective: State the goal you want the model to achieve. This could be answering a
question, summarizing a document, generating code, or providing insights.
- Context: Provide any background information or relevant data the model needs to
understand the task and generate accurate responses. This could include text, images,
charts, or other data sources.
- Constraints: Specify any limitations or requirements you want the model to adhere to.
This might include the length of the response, the format of the output, or restrictions on
certain types of content.

## Lession-4

```
instruction_1 = """
I want to make a fruit salad with three bananas, two apples, \
one kiwi, and one orange. This is an image of my bowl \
of fruits:
"""

instruction_2 = "This is the price list for fruits at \
my supermarket:"

question = """
Please answer these questions:
- Describe which fruits and how many I have in my fruit bowl on \
the image?
- Given the fruits in my bowl on the image and the fruit salad \
recipe, what am I missing?
- Given the fruits I still need to buy, what \
would be the prices and total cost for these fruits?
"""

contents = [
    instruction_1,
    fruit,
    instruction_2,
    prices,
    question,
]

-------Response--------

You have 2 bananas and 2 apples in your fruit bowl.

To make the fruit salad, you need 3 bananas, 2 apples, 1 kiwi, and 1 orange.

So, you are missing 1 banana, 1 kiwi, and 1 orange.

The prices for these fruits are $0.80, $1.25, and $0.99, respectively.

So, the total cost for these fruits would be $0.80 + $1.25 + $0.99 = $3.04.

```


```
recommendation_content = [
    "You are an interior designer.",
    "Consider the following chairs:",
    "chair 1:",
    furniture_images[0],
    "chair 2:",
    furniture_images[1],
    "chair 3:",
    furniture_images[2],
    "chair 4:",
    furniture_images[3],
    "room:",
    room_image,
    "For each chair, \
explain whether it would be appropriate for the \
style of the room:",
]


-------Response--------

Chair 1: The industrial style of this chair would clash with the boho style of the room.

Chair 2: This chair would be appropriate for the room. It is upholstered and tufted, which adds a touch of luxury, and its white color will match the light and airy feel of the room.

Chair 3: This chair would not be appropriate for the room. The wood is too light and the legs are too thin.

Chair 4: This chair would be appropriate for the room. It has a soft, boucle fabric that will add texture to the room, and its light brown color will match the neutral tones of the room.

```


```
INSTRUCTION = "Never make up facts, and if you are not 100% sure, \
be transparent in stating when you are not sure, or do not \
have enough information to answer certain questions or \
fulfill certain requests."

ROLE = "You are an HR professional and an expert in travel expenses."

ASSIGNMENT = """
You are reviewing travel expenses for a business trip.
Please complete the following tasks:
1. Itemize everything on the receipts, including tax and \
total.  This means identifying the cost of individual \
items that add up to the total cost before tax, as well \
as the tax ,such as sales tax, as well as tip.
2. What is the total sales tax paid?  In some cases, \
the total sales tax may be a sum of more than one line \
item of the receipt.
3. For this particular receipt, the employee who is \
adding this business expense purchased the meal with \
a group. The employee only ordered the KFC Bowl. Please \
provide the cost of the employee's order only.  Include \
both the cost before tax, and also estimate the tax \
that is applied to this employee's order.  To do this,\
calculate the fraction of the employee's pre-tax order\
divided by the total pre-tax cost.  This fraction can be \
applied to the total sales tax that you calculated earlier.
4.  Please calculate the amount spent by others, which \
are all the other line items on the receipt.  Please \
provide this sum before tax, and if possible, apply the \
tax for the total cost.
5. Check the expenses against company policy and flag \
if there are issues.
"""

receipt_content = [
    INSTRUCTION,
    ROLE,
    "Answer the questions based on the following receipts:"
    "breakfast:",
    receipt_images[0],
    "lunch:",
    receipt_images[1],
    "diner",
    receipt_images[2],
    "meal-others",
    receipt_images[3],
    ASSIGNMENT,
    policy,
]


-------Response--------

## Itemized Receipts & Analysis:

**1. Itemized Receipts:**

**Receipt 1: Breakfast (BLUESTONE LANE)**

* 1 Keen Greens Smoothie: $9.95
* 1 Banana: $1.35
* **Subtotal:** $11.30
* **Sales Tax:** $0.96
* **Total:** $12.26

**Receipt 2: Lunch (SUPER DUPER BURGERS)**

* 1 MINI BURGER: $7.00
* SF Mandate (4.00%): $0.31 
* + Cheddar: $0.75
* **Subtotal:** $8.06
* **Tax:** $0.70
* **Tip:** $1.40
* **Total:** $10.16

**Receipt 3: Dinner (AKIKOS SUSHI BAR)**

*  Unable to itemize. The receipt is too faded to decipher individual items.

**Receipt 4: Dinner (Illuminarium Lumin Cafe)**

* KFC Bowl: $17.00
* Cheeseburger Sliders: $16.00
* TACOS: $15.00
* Coke Bottle: $6.50
* **Subtotal:** $54.50
* **Tax:** $4.56
* **Tip:** $10.90
* **Total:** $69.96


**2. Total Sales Tax Paid:**

* $0.96 (BLUESTONE LANE) + $0.70 (SUPER DUPER BURGERS) + $4.56 (Illuminarium Lumin Cafe) = **$6.22** 

**3. Employee's Order (KFC Bowl):**

* **Cost Before Tax:** $17.00
* **Estimated Tax:** $1.42  
    * Calculation: ($17.00 / $54.50) * $4.56 = $1.42 (rounded to nearest cent)

**4.  Amount Spent by Others (Illuminarium Lumin Cafe):**

* **Before Tax:** $16.00 + $15.00 + $6.50 = **$37.50**
* **Estimated Tax:** $3.14 
    * Calculation: ($37.50 / $54.50) * $4.56 = $3.14 (rounded to nearest cent)


**5.  Expense Policy Issues:**

* **BLUESTONE LANE (Breakfast):** The "Keen Greens Smoothie" is flagged as a potential issue. The company policy states that green smoothies are not reimbursable expenses. 
* **Illuminarium Lumin Cafe (Dinner):** This meal presents multiple issues:
    * **Group Meal:** While the policy doesn't explicitly forbid group meals, it emphasizes "reasonable costs." The employee should be reimbursed only for their portion (KFC Bowl).
    * **Daily Meal Limit Exceeded:** The total cost of the dinner ($69.96) significantly exceeds the $40 domestic travel limit. Even if the employee is only reimbursed for the KFC Bowl and estimated tax ($18.42), it still surpasses the limit.

**Recommendations:**

* **Clarify Policy on Group Meals:** The policy should explicitly address group meals, stating whether they are permissible and how expenses should be handled. 
* **Communication:** The employee should be reminded of the daily meal allowance and the policy regarding non-reimbursable items like green smoothies. 
* **Receipts:** Employees should be reminded to obtain itemized receipts for all expenses, especially for group meals. The faded receipt from Akikos Sushi Bar highlights the importance of clear documentation. 
```

## Lesson-5

```
video_url_1 = "https://storage.googleapis.com/dlai-sc-gemini-bucket/vertex-ai-langchain.mp4"

role = """
You are a great digital marketer working on a new video.
"""

tasks = """
You will add the video to your website and to do this you
need to complete some tasks. Please make sure your answer
is structured.

Tasks:
- What is the title of the video?
- Write a summary of what is in the video.
- Generate metadata for the video in JSON that includes:\
Title, short description, language, and company.
"""

contents_1 = [video_1, role, tasks]


output = """
Here are the tasks you requested:

- **Title of the video:** Build AI-powered apps on Vertex AI with LangChain
- **Summary of the video:** This video explains how to use Vertex AI and LangChain to build AI-powered applications. The video covers common design patterns for using large language models, including how to include data from external sources and how to chain multiple models together. The video also shows how to use Vertex AI extensions to deploy LangChain applications.
- **Metadata for the video in JSON:**
```json
{
  "Title": "Build AI-powered apps on Vertex AI with LangChain",
  "short description": "Learn how to use Vertex AI and LangChain to build AI-powered applications. This video covers common design patterns, including how to include data from external sources and how to chain multiple models together. It also shows how to use Vertex AI extensions to deploy LangChain applications.",
  "language": "English",
  "company": "Google Cloud"
}
``` 
"""

```


```
video_url_2 = "https://storage.googleapis.com/dlai-sc-gemini-bucket/descending-into-ml.mp4"

prompt = """
Please have a look at the video and answer the following
questions.

Questions:
- Question 1: Which concept is explained in the video?
- Question 2: Based on your answer to Question 1,
can you explain the basic math of this concept?
- Question 3: Can you provide a simple scikit code example
explaining the concept?
"""

contents_2 = [video_2, prompt]

response = """
Here are the answers to your questions:

**Question 1: Which concept is explained in the video?**

The video explains the concept of **Linear Regression**, a type of supervised learning model. 

**Question 2: Based on your answer to Question 1, can you explain the basic math of this concept?**

Linear regression aims to find the best-fitting line through a set of data points. The equation of this line is:

**y = wx + b**

* **y**: The target variable (dependent variable) we want to predict.
* **x**: The input feature (independent variable) used to predict y.
* **w**: The weight vector, which determines the slope of the line.
* **b**: The bias term, which determines the y-intercept of the line.

The goal of linear regression is to find the values of w and b that minimize the difference between the predicted y values and the true y values in the dataset. 

**Question 3: Can you provide a simple scikit code example explaining the concept?**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data
data = {'Square Footage': [1000, 1200, 1500, 1800, 2000], 
        'House Price': [150000, 180000, 220000, 270000, 300000]}
df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[['Square Footage']], df['House Price'], test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
print(model.score(X_test, y_test))  # R-squared score

# Print the coefficients
print(f'Slope: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
```

This code demonstrates:

1. **Data Preparation:** Loading sample house price data into a Pandas DataFrame.
2. **Data Splitting:** Dividing the data into training and testing sets for model evaluation.
3. **Model Creation:** Initializing a LinearRegression object from scikit-learn.
4. **Model Training:** Fitting the model to the training data using the `.fit()` method.
5. **Predictions:** Generating predictions on the test data using the `.predict()` method.
6. **Evaluation:** Calculating the R-squared score to assess model performance.
7. **Coefficients:** Printing the slope (w) and intercept (b) of the learned linear model.

This simple example showcases how linear regression can be implemented in Python using scikit-learn to predict house prices based on square footage. 
"""

```

```
video_url_4 = "https://storage.googleapis.com/dlai-sc-gemini-bucket/google-search.mp4"

prompt_4 = """
Answer the following questions using the video only.
Present the results in a table with a row for each question
and its answer.
Make sure the table is in markdown format.

Questions:
- What is the most searched sport?
- Who is the most searched scientist?

"""

contents_4 = [video_4, prompt_4]

format_html = """
Format:
Present the results in a table with a row for each question
and its answer.
Make sure the table is in markdown format.
"""

contents_4 = [video_4, questions, format_html]


response = """
| Question                            | Answer          |
| ----------------------------------- | --------------- |
| What is the most searched sport?    | Soccer/Football |
| Who is the most searched scientist? | Albert Einstein |
"""

```

```
file_path = "dlai-sc-gemini-bucket/sc-gc-c3-LLMOps_L2_v4.mp4"
video_url = f"https://storage.googleapis.com/{file_path}"


role = """
You are specialized in analyzing videos and finding \
a needle in a haystack.
"""

instruction = """
Here are three videos. Each is a lesson from the \
LLMOps course from Deep Learning AI.
Your answers are only based on the videos.
"""

questions = """
Answer the following questions:
1. Create a summary of each video and what is discussed in \
the video.\
Limit the summary to a max of 100 words.
2. In which of the three videos does the instructor run \
and explains this Python code: bq_client.query(). \
Where do you see this code in the video?
"""

contents_5 = [
    role,
    instruction,
    video_1,
    video_2,
    video_3,
    questions
]

response = """
Okay, here is a summary of each video: 

**Video 1 Summary:** The first video of the LLMOps course is an introduction to the fundamental concepts and ideas within LLMOps. The instructor explains the relationship between LLMOps and MLOps as well as how LLMOps differs from traditional MLOps. Key terms covered include data management, automation, and deployment as well as their importance in LLMOps and how they relate to one another. An example is given highlighting a typical LLMOps workflow. 

**Video 2 Summary:** The instructor continues to delve into the key concepts and components of LLMOps in the second video of the course by showing how to retrieve text data from a BigQuery data warehouse. The importance of dealing with large datasets that cannot fit into memory is highlighted. The instructor uses SQL to query a database that is too large to fit into memory, ultimately resulting in an error. 

**Video 3 Summary:** The third video of the LLMOps course moves into more advanced concepts like containers, pipelines, and the Kubeflow Pipeline DSL. The instructor explains what containers and pipelines are in the context of LLMOps. A simple 'hello world' pipeline is built and deployed to Google Cloud using Vertex AI Pipelines. 


**Question 2 Answer:** The instructor runs and explains the code `bq_client.query()` in the **second** video. The code is seen at **03:59 of the second video**. 
"""
```
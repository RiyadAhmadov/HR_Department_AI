import uvicorn
from fastapi import FastAPI, Query
import pickle
import pandas as pd
import warnings
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Create a FastAPI app instance
app = FastAPI()


# Load the pickled model using a relative file path
with open('gb_model.pkl', "rb") as model_file:
    model = pickle.load(model_file)

# Define the HTML content
html_content = """
<!DOCTYPE html>
<html>
<style>
    body {
        background-image: url(https://images.unsplash.com/photo-1504670073073-6123e39e0754?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80);
        background-size: cover;
        text-align: center; /* Center-align text within the body */
    }
    
    form {
        text-align: left;
    }

    h1 {
        font-size: 28px; /* Change the font size for the heading */
        color: #333; /* Change the text color for the heading */
    }

    label {
        font-size: 16px; /* Change the font size for labels */
        color: #555; /* Change the text color for labels */
    }

    input {
        font-size: 16px; /* Change the font size for input fields */
        padding: 5px; /* Add padding to input fields */
        margin: 5px 0; /* Add margin to input fields */
        width: 100%; /* Make input fields 100% width of their container */
    }

    input[type="submit"] {
        background-color: #007BFF; /* Change the background color for the submit button */
        color: #fff; /* Change the text color for the submit button */
        font-size: 18px; /* Change the font size for the submit button */
        padding: 10px 20px; /* Add padding to the submit button */
        cursor: pointer;
    }

    input[type="submit"]:hover {
        background-color: #0056b3; /* Change the background color on hover */
    }


    .header {
        background-color: rgba(255, 255, 255, 0.5);
        padding: 2px;
        border: 10px solid rgba(255, 255, 255, 0.5);
    }

    .form-container {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 20px;
        margin: 20px auto; /* Add margin to create space */
        width: 50%;
        border-radius: 10px;
    }
    .header h1 
    {
            color: black; /* Set text color to white */
            font-size: 30px;
            font-family: "Verdana", sans-serif;
    }

    h2 {
        font-size: 20px; /* Change the font size for the result heading */
        color: #333; /* Change the text color for the result heading */
    }

    p {
        font-size: 18px; /* Change the font size for the result text */
        color: #333; /* Change the text color for the result text */
    }
</style>
<head>
    <title>Resume Quality Prediction</title>
</head>
<body>
    <div class="header">
        <h1>Code Academy Final Project</h1>
    </div>
    <div class="form-container">
    <h1>Resume Quality Prediction</h1>
    <form action="/predict/" method="get">
        <label for="received_callback">Received Callback (0 or 1):</label>
        <input type="number" id="received_callback" name="received_callback" required><br>

        <label for="firstname">First Name:</label>
        <input type="text" id="firstname" name="firstname" required><br>

        <label for="race">Race (White or Black):</label>
        <input type="text" id="race" name="race" required><br>

        <label for="gender">Gender (0 or 1):</label>
        <input type="number" id="gender" name="gender" required><br>

        <label for="years_college">Years in College:</label>
        <input type="number" id="years_college" name="years_college" required><br>

        <label for="college_degree">College Degree (0 or 1):</label>
        <input type="number" id="college_degree" name="college_degree" required><br>

        <label for="honors">Honors (0 or 1):</label>
        <input type="number" id="honors" name="honors" required><br>

        <label for="years_experience">Years of Experience:</label>
        <input type="number" id="years_experience" name="years_experience" required><br>

        <label for="worked_during_school">Worked During School (0 or 1):</label>
        <input type="number" id="worked_during_school" name="worked_during_school" required><br>

        <label for="computer_skills">Computer Skills  (0 or 1):</label>
        <input type="number" id="computer_skills" name="computer_skills" required><br>

        <label for="special_skills">Special Skills  (0 or 1):</label>
        <input type="number" id="special_skills" name="special_skills" required><br>

        <label for="volunteer">Volunteer (0 or 1):</label>
        <input type="number" id="volunteer" name="volunteer" required><br>

        <label for="military">Military (0 or 1):</label>
        <input type="number" id="military" name="military" required><br>

        <label for="employment_holes">Employment Holes  (0 or 1):</label>
        <input type="number" id="employment_holes" name="employment_holes" required><br>

        <label for="has_email_address">Has Email Address (0 or 1):</label>
        <input type="number" id="has_email_address" name="has_email_address" required><br>

        <input type="submit" value="Predict Resume Quality">
    </form>
    <h2>Prediction Result:</h2>
    <p id="prediction_result"></p>
    </div>
    <script>
        const form = document.querySelector('form');
        const predictionResult = document.getElementById('prediction_result');
        
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            
            const formData = new FormData(form);
            
            const response = await fetch('/predict/?' + new URLSearchParams(formData).toString());
            const data = await response.json();
            
            predictionResult.textContent = data['prediction'];
        });
    </script>
</body>
</html>
"""

# Define the endpoint to serve the HTML content
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=html_content)

# Define the endpoint to make predictions
@app.get("/predict/")
async def predict(
    received_callback: int = Query(..., description="Received Callback"),
    firstname: str = Query(..., description="First Name"),
    race: str = Query(..., description="Race"),
    gender: int = Query(..., description="Gender"),
    years_college: int = Query(..., description="Years in College"),
    college_degree: int = Query(..., description="College Degree"),
    honors: int = Query(..., description="Honors"),
    years_experience: int = Query(..., description="Years of Experience"),
    worked_during_school: int = Query(..., description="Worked During School"),
    computer_skills: int = Query(..., description="Computer Skills"),
    special_skills: int = Query(..., description="Special Skills"),
    volunteer: int = Query(..., description="Volunteer"),
    military: int = Query(..., description="Military"),
    employment_holes: int = Query(..., description="Employment Holes"),
    has_email_address: int = Query(..., description="Has Email Address"),
):
    # Create a DataFrame from the input data
    data = pd.DataFrame({
        'received_callback': [received_callback],
        'firstname': [firstname],
        'race': [race],
        'gender': [gender],
        'years_college': [years_college],
        'college_degree': [college_degree],
        'honors': [honors],
        'years_experience': [years_experience],
        'worked_during_school': [worked_during_school],
        'computer_skills': [computer_skills],
        'special_skills': [special_skills],
        'volunteer': [volunteer],
        'military': [military],
        'employment_holes': [employment_holes],
        'has_email_address': [has_email_address],
    })

    # Calculate new columns
    data['milvol'] = data.apply(lambda row: 1 if row['volunteer'] == 1 and row['military'] == 1 else 0, axis=1)
    data['avg_milwork'] = (data['military'] + data['worked_during_school']) / 2
    data['mltp_experspecskill'] = data['years_experience'] * data['special_skills']
    data['positive_col'] = data[['honors',"computer_skills","volunteer","military"]].sum(axis = 1)
    # Make predictions using the pre-trained model
    predictions = model.predict(data[['worked_during_school', 'computer_skills', 'volunteer', 'military', 'employment_holes', 'has_email_address','milvol','positive_col','avg_milwork', 'mltp_experspecskill']])

    # Interpret predictions
    prediction_result = f"{firstname} , quality of your CV is bad !!! (Please Change CV)" if predictions[0] == 0 else f"{firstname} , quality of your CV is good !!!"

    return {"prediction": prediction_result}

# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5002,
        log_level="debug",
    )

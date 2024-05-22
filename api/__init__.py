from fastapi import FastAPI, File, UploadFile
import os
from .docanswer import DocAnswer


# Create an instance of FastAPI
app = FastAPI()


# Define a simple route
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


# Define a route for file upload
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Create a directory to store uploaded files if it doesn't exist
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")

    # Save the uploaded file to the 'uploaded_files' directory
    file_location = f"uploaded_files/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())

    # Process the uploaded file (for demonstration, let's just return the file details)
    processed_data = await process_file(file_location)

    # Return the processed data
    return processed_data


async def process_file(file_location):
    # Get question list
    rag_doc = DocAnswer(file_location, model="gpt-4o")
    answers = await rag_doc.answer_file()
    # answers2 = await rag_doc.answer_file(questions[10:20])
    # answers3 = await rag_doc.answer_file(questions[20:])

    
    
    # answers = []

    return answers


# Run the FastAPI app using uvicorn server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
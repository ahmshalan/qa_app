from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import os
from .docanswer import DocAnswer


# Create an instance of FastAPI

app = FastAPI(debug=True, title="qa", description="question answering app")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def main():
    return RedirectResponse(url="/docs/")

# Define a route for file upload
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Create a directory to store uploaded files if it doesn't exist
    if not os.path.exists("/tmp/uploaded_files"):
        os.makedirs("/tmp/uploaded_files")

    # Save the uploaded file to the 'uploaded_files' directory
    file_location = f"/tmp/uploaded_files/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())

    print("Start PRocessing ....")

    # Process the uploaded file (for demonstration, let's just return the file details)
    processed_data = await process_file(file_location)

    # Return the processed data
    return processed_data


async def process_file(file_location):
    # Get question list
    rag_doc = DocAnswer(file_location, model="gpt-4o")
    answers = await rag_doc.answer_file()
    return answers



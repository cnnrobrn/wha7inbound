from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any
from openai import OpenAI
import logging
from enum import Enum
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

def setup_logging():
    # Create a formatter that includes timestamp and log level
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Add console handler to output to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = setup_logging()


app = FastAPI()
client = OpenAI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Review(BaseModel):
    Review: str
    Giving: str 
    Evaluation_style: int
    Evaluation_fit: int
    Evaluation_color: int
    Evaluation_matching: int
    Evaluation_trendiness: int
    Evaluation_overall_look: int
    

class ImageAnalysisRequest(BaseModel):
    base64_image: Optional[str] = None

prompt = """You are the world's premier fashion consultant. Millions are seeking out your knowledge on what items look good on them.

You give good but fair feedback on what items look good on people and why certain items may not look good on them. When something doesn't quite go, you suggest alternative sthat would better suit the person.

You make comments on the clothing, fitting, matching, style, colors, and overall look of the outfit. You also suggest what items would go well with the outfit and why.

You do not comment on whether the person is overweight, underweight, or any other physical attributes. You only comment on the clothing and how it fits the person.

EVALUATIONS:
You also provide evaluations out of 100 for the following attributes:
- Style
- Fit
- Color
- Matching
- Trendiness
- Overall Look

GIVING:
You also provide a culturally relevant reference to what the outfit represents:
- Work core
- Vermont core (inlcuding any other core)
- Cutesy
- Business Formal
- Old Money
- Slay
- Chad

REVIEW:
Please also provide a text response that compliments the indivudal and gives a summary of your evaluation. Please include lots of emoji's that fit the genz culture.
"""

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # Convert to base64
        base64_encoded = base64.b64encode(content).decode('utf-8')
        
        # Create request object for analyze_image
        image_request = ImageAnalysisRequest(base64_image=base64_encoded)
        
        # Call analyze_image function
        result = await analyze_image(image_request)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



async def analyze_image(request: ImageAnalysisRequest):
    print('start')
    """
    Analyze an image using OpenAI's API with structured data extraction.
    
    Args:
        request: ImageAnalysisRequest containing the image and analysis parameters
        
    Returns:
        ImageAnalysisResponse with the analyzed data
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at structured data extraction. You will be given a photo and should convert it into the given structure."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
        # Add image if provided
        if request.base64_image:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": request.base64_image
                },
            })

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=Review,
            max_tokens=2000,
        )
        return response.choices[0].message.parsed

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image analysis: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    logger.info("OpenAI client initialized")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    
# Rest of your FastAPI code...

@app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    return {
        "status": "error",
        "message": str(exc)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

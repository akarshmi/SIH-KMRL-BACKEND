import os
import aiohttp
import cloudinary.uploader
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from api.app.config import supabase
from api.app.schemas.models import URLRequest, SUMMARYRequest, ListDocsRequest, compliancesRequest, searchRequest, SearchResponse
import json
from fastapi import Request
import logging
import traceback
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional ML imports
try:
    import torch
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    ML_AVAILABLE = False
    torch = None
    SentenceTransformer = None
    model = None

router = APIRouter()

UPLOAD_DIR = "./temp"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/url")
async def receive_url(request: URLRequest):
    logger.info(f"Received URL request: {request.url}")
    
    try:
        file_url = request.url
        filename = file_url.split("/")[-1]
        file_location = os.path.join(UPLOAD_DIR, filename)
        
        logger.info(f"Downloading file to: {file_location}")

        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status != 200:
                    logger.error(f"Download failed with status: {resp.status}")
                    raise HTTPException(status_code=400, detail="Download failed")
                content = await resp.read()
                with open(file_location, "wb") as f:
                    f.write(content)

        try:
            logger.info("Uploading to Cloudinary...")
            # Upload file path instead of content for better handling
            upload_result = cloudinary.uploader.upload(
                file_location,  # Use file path instead of content
                resource_type="raw",
                use_filename=True,
                unique_filename=True,  # Changed to True to avoid conflicts
                folder="documents"  # Added folder for organization
            )
            
            logger.info(f"Cloudinary response: {upload_result}")
            
            # Get the secure URL (always prefer secure_url)
            cloudinary_url = upload_result.get("secure_url")
            if not cloudinary_url:
                logger.error("No secure_url in Cloudinary response")
                raise HTTPException(status_code=500, detail="Failed to get Cloudinary URL")

            logger.info(f"Cloudinary URL generated: {cloudinary_url}")

            logger.info("Fetching department...")
            dept_resp = supabase.table("departments").select("dept_id").eq("name", request.dept_name).execute()
            if not dept_resp.data:
                logger.error(f"Department not found: {request.dept_name}")
                raise HTTPException(status_code=400, detail="Department not found")
            dept_id = dept_resp.data[0]["dept_id"]

            logger.info("Inserting document...")
            doc_resp = supabase.table("documents").insert({
                "title": filename,
                "department": dept_id,
                "url": cloudinary_url,
                "medium": "url",
                "priority": request.priority,
            }).execute()
            
            inserted_doc = doc_resp.data[0] if doc_resp.data else None

            if inserted_doc:
                doc_id = inserted_doc["doc_id"]
                logger.info(f"Document inserted with ID: {doc_id}")
                
                # Insert summary
                supabase.table("summaries").insert({
                    "doc_id": doc_id,
                    "content": ""
                }).execute()
                
                # Insert transaction
                supabase.table("transexions").insert({
                    "from_user": request.user_id,
                    "to_department": dept_id,
                    "doc_id": doc_id
                }).execute()

        finally:
            # Clean up temporary file
            if os.path.exists(file_location):
                os.remove(file_location)

        return {
            "document": doc_resp.data,
            "filename": filename,
            "processed": "",
            "cloudinary_url": cloudinary_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in receive_url: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/file")
async def receive_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    dept_name: str = Form(...),
    priority: str = Form(...)
):
    logger.info(f"Received file upload: {file.filename}")
    logger.info(f"User ID: {user_id}, Dept: {dept_name}, Priority: {priority}")
    logger.info(f"File content type: {file.content_type}")
    logger.info(f"File size: {file.size}")
    
    temp_file_path = None
    
    try:
        # Validate inputs
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        if not dept_name:
            raise HTTPException(status_code=400, detail="Department name required")
        if not priority:
            raise HTTPException(status_code=400, detail="Priority required")
        
        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/msword", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/bmp",
            "image/webp",
        ]
        
        # Also check file extension as backup
        allowed_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        file_extension = Path(file.filename).suffix.lower()
        
        if file.content_type not in allowed_types and file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Allowed types: PDF, Word, Excel, Images"
            )
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large: {file.size} bytes. Maximum allowed: {max_size} bytes (10MB)"
            )
            
        # Create temporary file with proper extension
        suffix = file_extension if file_extension else '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            
        logger.info(f"Created temporary file: {temp_file_path}")
        
        # Read and save file content
        content = await file.read()
        logger.info(f"Read file content: {len(content)} bytes")
        
        with open(temp_file_path, "wb") as f:
            f.write(content)

        logger.info("Checking department...")
        # Check if department exists
        dept_resp = supabase.table("departments").select("dept_id").eq("name", dept_name).execute()
        if not dept_resp.data:
            logger.error(f"Department not found: {dept_name}")
            raise HTTPException(status_code=400, detail=f"Department '{dept_name}' not found")
        dept_id = dept_resp.data[0]["dept_id"]
        logger.info(f"Found department ID: {dept_id}")

        logger.info("Uploading to Cloudinary...")
        
        # Determine resource type based on file type
        resource_type = "auto"  # Let Cloudinary auto-detect
        if file.content_type and file.content_type.startswith('image/'):
            resource_type = "image"
        elif file.content_type == "application/pdf":
            resource_type = "image"  # PDFs are handled as images in Cloudinary
        else:
            resource_type = "raw"  # For documents
        
        # Upload to Cloudinary using file path
        upload_result = cloudinary.uploader.upload(
            temp_file_path,
            resource_type="raw",
            use_filename=True,
            unique_filename=True,  # Ensure unique filename
            folder="documents", 
                access_mode="public",  # Make it publicly accessible

              # Organize in folder
            # Add file type transformation for better handling
            format=file_extension[1:] if file_extension and len(file_extension) > 1 else None
        )
        
        logger.info(f"Cloudinary upload result: {upload_result}")
        
        # Get the secure URL
        cloudinary_url = upload_result.get("secure_url")
        if not cloudinary_url:
            logger.error("No secure_url found in Cloudinary response")
            logger.error(f"Full Cloudinary response: {upload_result}")
            raise HTTPException(status_code=500, detail="Failed to get Cloudinary secure URL")

        logger.info(f"Cloudinary secure URL: {cloudinary_url}")
        
        # Test the URL accessibility (optional but helpful for debugging)
        try:
            import requests
            test_response = requests.head(cloudinary_url, timeout=5)
            logger.info(f"URL accessibility test: {test_response.status_code}")
            if test_response.status_code != 200:
                logger.warning(f"Cloudinary URL may not be accessible: {test_response.status_code}")
        except Exception as test_error:
            logger.warning(f"Could not test URL accessibility: {test_error}")

        logger.info("Inserting document record...")
        # Insert document
        doc_resp = supabase.table("documents").insert({
            "title": file.filename,
            "department": dept_id,
            "url": cloudinary_url,
            "medium": "direct file",
            "priority": priority,
        }).execute()
        
        logger.info(f"Document insert response: {doc_resp}")
        
        inserted_doc = doc_resp.data[0] if doc_resp.data else None
        if not inserted_doc:
            raise HTTPException(status_code=500, detail="Failed to insert document")

        doc_id = inserted_doc["doc_id"]
        logger.info(f"Document inserted with ID: {doc_id}")
        
        logger.info("Inserting summary...")
        # Insert summary
        summary_resp = supabase.table("summaries").insert({
            "doc_id": doc_id,
            "content": ""
        }).execute()
        logger.info(f"Summary insert response: {summary_resp}")
        
        logger.info("Inserting transaction...")
        # Insert transaction
        trans_resp = supabase.table("transexions").insert({
            "from_user": user_id,
            "to_department": dept_id,
            "doc_id": doc_id
        }).execute()
        logger.info(f"Transaction insert response: {trans_resp}")

        logger.info("Upload completed successfully")
        return {
            "document": doc_resp.data,
            "filename": file.filename,
            "processed": "",
            "cloudinary_url": cloudinary_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in receive_file: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Always clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

# Add a test endpoint to verify Cloudinary configuration
@router.get("/test-cloudinary-config")
async def test_cloudinary_config():
    try:
        import cloudinary
        
        # Test configuration
        config = cloudinary.config()
        logger.info(f"Cloudinary config: cloud_name={config.cloud_name}")
        
        if not config.cloud_name or not config.api_key or not config.api_secret:
            return {
                "status": "error",
                "message": "Cloudinary configuration incomplete",
                "config": {
                    "cloud_name": bool(config.cloud_name),
                    "api_key": bool(config.api_key),
                    "api_secret": bool(config.api_secret)
                }
            }
        
        # Test upload with a simple text file
        test_result = cloudinary.uploader.upload(
            "data:text/plain;base64,SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            resource_type="raw",
            public_id="test_upload",
            folder="test"
        )
        
        # Clean up test file
        try:
            cloudinary.uploader.destroy("test/test_upload", resource_type="raw")
        except:
            pass
        
        return {
            "status": "success",
            "message": "Cloudinary configuration is working",
            "test_url": test_result.get("secure_url"),
            "cloud_name": config.cloud_name
        }
        
    except Exception as e:
        logger.error(f"Cloudinary configuration test failed: {str(e)}")
        return {
            "status": "error", 
            "message": f"Cloudinary test failed: {str(e)}"
        }

@router.get("/summary")
async def summary(doc_id: str):
    try:
        logger.info(f"Fetching summary for doc_id: {doc_id}")
        response = supabase.table("summaries").select("content") \
            .eq("doc_id", doc_id).execute()
        if response.data:
            return {"summary": response.data[0]["content"]}
        return {"error": "No summary found"}
    except Exception as e:
        logger.error(f"Error fetching summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary: {str(e)}")

@router.get("/listdocs")
async def listdocs(request: Request, user_id: str):
    try:
        logger.info(f"Listing docs for user: {user_id}")
        redis_conn = request.app.state.redis
        cache_key = f"user_docs:{user_id}"

        # Check if cached
        cached_data = await redis_conn.get(cache_key)
        if cached_data:
            return {"data": json.loads(cached_data), "cached": True}

        # Fetch department ID
        dept_resp = supabase.table("users").select("department").eq("id", user_id).execute()
        if not dept_resp.data:
            raise HTTPException(status_code=404, detail="User not found")

        dept_id = dept_resp.data[0]["department"]

        # Fetch documents
        response = supabase.table("documents").select("*").eq("department", dept_id).execute()
        if not response.data:
            return {"data": [], "cached": False}

        # Cache result for 60 seconds
        await redis_conn.set(cache_key, json.dumps(response.data), ex=60)

        return {"data": response.data, "cached": False}
    except Exception as e:
        logger.error(f"Error listing docs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@router.get("/compliances")
async def compliances(doc_id: str):
    try:
        logger.info(f"Fetching compliances for doc_id: {doc_id}")
        response = supabase.table("compliance").select("*").eq("doc_id", doc_id).execute()
        if response.data:
            return {"data": response.data}
        return {"data": [], "message": "No compliances found"}
    except Exception as e:
        logger.error(f"Error fetching compliances: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch compliances: {str(e)}")

@router.get("/search", response_model=list[SearchResponse])
async def search(query: str = Query(...), top_k: int = 5):
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Search functionality not available. ML dependencies (torch, sentence-transformers) are not installed."
        )
    
    try:
        logger.info(f"Searching for: {query}")
        # Embed query
        with torch.inference_mode():
            q = model.encode([query], normalize_embeddings=True).tolist()[0]

        # Run pgvector similarity
        sql = f"""
            select doc_id, chunk_id, content,
                   1 - (embedding <-> '{q}') as score
            from document_chunks
            order by embedding <-> '{q}'
            limit {top_k};
        """

        res = supabase.rpc("exec_sql", {"query": sql}).execute()
        return res.data
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Test endpoint to check if the API is working
@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Document API is running"}

# Test endpoint to check database connectivity
@router.get("/test-db")
async def test_database():
    try:
        # Test Supabase connection
        response = supabase.table("departments").select("count").execute()
        return {"status": "database_connected", "message": "Database connection successful"}
    except Exception as e:
        logger.error(f"Database test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Test endpoint to check Cloudinary
@router.get("/test-cloudinary")
async def test_cloudinary():
    try:
        # Test Cloudinary configuration
        import cloudinary
        config = cloudinary.config()
        return {
            "status": "cloudinary_configured", 
            "cloud_name": config.cloud_name,
            "message": "Cloudinary configuration found"
        }
    except Exception as e:
        logger.error(f"Cloudinary test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cloudinary configuration error: {str(e)}")
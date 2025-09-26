import os
import aiohttp
import cloudinary.uploader
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from api.app.config import supabase
from api.app.schemas.models import URLRequest, SUMMARYRequest, ListDocsRequest, compliancesRequest, searchRequest, SearchResponse
# from nlpPipelne.ProcessPipeline import process_file
# from nlpPipelne.stages.EmbedIndex import search
import json
from fastapi import Request

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

@router.post("/url")
async def receive_url(request: URLRequest):
    file_url = request.url
    filename = file_url.split("/")[-1]
    file_location = os.path.join(UPLOAD_DIR, filename)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail="Download failed")
            content = await resp.read()
            with open(file_location, "wb") as f:
                f.write(content)

    try:
        # Upload to Cloudinary with proper configuration
        upload_result = cloudinary.uploader.upload(
            content, 
            resource_type="auto",
            use_filename=True,
            unique_filename=False
        )
        
        # Get the correct URL field
        cloudinary_url = upload_result.get("secure_url") or upload_result.get("url")
        if not cloudinary_url:
            raise HTTPException(status_code=500, detail="Failed to get Cloudinary URL")

        dept_resp = supabase.table("departments").select("dept_id").eq("name", request.dept_name).execute()
        if not dept_resp.data:
            raise HTTPException(status_code=400, detail="Department not found")
        dept_id = dept_resp.data[0]["dept_id"]

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
            # supabase.table("summaries").insert({
            #     "doc_id": doc_id,
            #     "content": output["doc_summary"]
            # }).execute()
            supabase.table("summaries").insert({
                "doc_id": doc_id,
                "content": ""
            }).execute()
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

@router.post("/file")
async def receive_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    dept_name: str = Form(...),
    priority: str = Form(...)
):
    file_location = os.path.join(UPLOAD_DIR, filename)
    content = await file.read()
    with open(file_location, "wb") as f:
        f.write(content)

    try:
        # Upload to Cloudinary with proper configuration
        upload_result = cloudinary.uploader.upload(
            content, 
            resource_type="auto",
            use_filename=True,
            unique_filename=False,
            folder="documents"  # Optional: organize files in folders
        )
        
        # Get the correct URL field
        cloudinary_url = upload_result.get("secure_url") or upload_result.get("url")
        if not cloudinary_url:
            raise HTTPException(status_code=500, detail="Failed to get Cloudinary URL")

        dept_resp = supabase.table("departments").select("dept_id").eq("name", dept_name).execute()
        if not dept_resp.data:
            raise HTTPException(status_code=400, detail="Department not found")
        dept_id = dept_resp.data[0]["dept_id"]

        doc_resp = supabase.table("documents").insert({
            "title": file.filename,
            "department": dept_id,
            "url": cloudinary_url,
            "medium": "direct file",
            "priority": priority,
        }).execute()
        
        inserted_doc = doc_resp.data[0] if doc_resp.data else None

        if inserted_doc:
            doc_id = inserted_doc["doc_id"]
            # supabase.table("summaries").insert({
            #     "doc_id": doc_id,
            #     "content": output.get("doc_summary", "")
            # }).execute()
            supabase.table("summaries").insert({
                "doc_id": doc_id,
                "content": ""
            }).execute()
            supabase.table("transexions").insert({
                "from_user": user_id,
                "to_department": dept_id,
                "doc_id": doc_id
            }).execute()

    except Exception as e:
        # Clean up on error
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        # Always clean up temporary file
        if os.path.exists(file_location):
            os.remove(file_location)

    return {
        "document": doc_resp.data,
        "filename": file.filename,
        "processed": "",
        "cloudinary_url": cloudinary_url
    }

@router.get("/summary")
async def summary(doc_id: str):
    try:
        response = supabase.table("summaries").select("content") \
            .eq("doc_id", doc_id).execute()
        if response.data:
            return {"summary": response.data[0]["content"]}
        return {"error": "No summary found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary: {str(e)}")

@router.get("/listdocs")
async def listdocs(request: Request, user_id: str):
    try:
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
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@router.get("/compliances")
async def compliances(doc_id: str):
    try:
        response = supabase.table("compliance").select("*").eq("doc_id", doc_id).execute()
        if response.data:
            return {"data": response.data}
        return {"data": [], "message": "No compliances found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch compliances: {str(e)}")

@router.get("/search", response_model=list[SearchResponse])
async def search(query: str = Query(...), top_k: int = 5):
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Search functionality not available. ML dependencies (torch, sentence-transformers) are not installed."
        )
    
    try:
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
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Additional utility endpoints

@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its related data"""
    try:
        # Delete summaries first (foreign key constraint)
        supabase.table("summaries").delete().eq("doc_id", doc_id).execute()
        
        # Delete compliance records
        supabase.table("compliance").delete().eq("doc_id", doc_id).execute()
        
        # Delete transactions
        supabase.table("transexions").delete().eq("doc_id", doc_id).execute()
        
        # Delete document chunks if they exist
        supabase.table("document_chunks").delete().eq("doc_id", doc_id).execute()
        
        # Finally delete the document
        response = supabase.table("documents").delete().eq("doc_id", doc_id).execute()
        
        if response.data:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get document details by ID"""
    try:
        response = supabase.table("documents").select("*").eq("doc_id", doc_id).execute()
        if response.data:
            return {"data": response.data[0]}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {str(e)}")

@router.put("/document/{doc_id}")
async def update_document(doc_id: str, title: str = Form(None), priority: str = Form(None)):
    """Update document metadata"""
    try:
        update_data = {}
        if title:
            update_data["title"] = title
        if priority:
            update_data["priority"] = priority
            
        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")
            
        response = supabase.table("documents").update(update_data).eq("doc_id", doc_id).execute()
        
        if response.data:
            return {"message": "Document updated successfully", "data": response.data[0]}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")
"""ODRA FastAPI Application."""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db import init_db, SessionLocal, AuditJob
from app.api import health, audit, ingest
from app.services.auditor import AuditorPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Background task for audit processing
_audit_processor_task = None


async def process_audit_queue():
    """Background task to process audit jobs from queue."""
    logger.info("🟢 Audit queue processor started")
    
    while True:
        try:
            from app.services.task_queue import task_queue_service
            
            # Get next audit task
            task = await task_queue_service.dequeue("audit")
            
            if task:
                job_id, payload = task
                logger.info(f"🔄 Processing audit job: {job_id}")
                
                try:
                    db = SessionLocal()
                    job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
                    
                    if job:
                        job.status = "processing"
                        db.commit()
                        
                        # Run audit
                        planner = AuditorPlanner(
                            goal=payload.get("goal"),
                            scope=payload.get("scope", "")
                        )
                        
                        result = await planner.run_audit(job_id)
                        
                        # Update job with results
                        job.status = "completed"
                        job.progress = 100.0
                        job.results = result
                        db.commit()
                        
                        logger.info(f"✅ Audit job {job_id} completed")
                    
                    db.close()
                
                except Exception as e:
                    logger.error(f"❌ Audit job {job_id} failed: {e}", exc_info=True)
                    db = SessionLocal()
                    job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
                    if job:
                        job.status = "failed"
                        db.commit()
                    db.close()
            else:
                # No task, wait a bit before checking again
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Audit queue processor error: {e}", exc_info=True)
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    logger.info("Starting ODRA Backend...")
    await init_db()
    logger.info("Database initialized")
    
    # Start audit processor
    global _audit_processor_task
    _audit_processor_task = asyncio.create_task(process_audit_queue())
    logger.info("Audit processor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ODRA Backend...")
    if _audit_processor_task:
        _audit_processor_task.cancel()
        try:
            await _audit_processor_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ODRA - Outcome-Driven RAG Auditor",
    description="Semantic document processing and RAG-powered audit reports",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(audit.router, prefix="/audit", tags=["audit"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ODRA Backend",
        "status": "operational",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

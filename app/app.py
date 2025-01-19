from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Import your existing classes
from .analyzer import (
    VideoParameterAnalyzer,
    CompanyInfo,
    CompetitorAnalyzer,
    YouTubeAnalyzer
)

app = FastAPI(title="Video Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CompanyRequest(BaseModel):
    name: str
    domain: str
    description: str
    ad_objective: str

class VideoAnalysisResponse(BaseModel):
    competitors: List[str]
    video_analysis: List[Dict]
    top_parameters: List[tuple]
    top_impact_metrics: List[tuple]
    insights: List[str]
    sentiment_analysis: Dict[str, float]
    transcript_length: int

@app.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_videos(company_data: CompanyRequest):
    try:
        # Initialize analyzers
        competitor_analyzer = CompetitorAnalyzer()
        youtube_analyzer = YouTubeAnalyzer()
        video_analyzer = VideoParameterAnalyzer()
        
        # Create company info instance
        company_info = CompanyInfo(
            name=company_data.name,
            domain=company_data.domain,
            description=company_data.description,
            ad_objective=company_data.ad_objective
        )
        
        # Get competitors
        competitors = competitor_analyzer.get_competitors(company_info)
        
        # Search for videos
        videos_found = youtube_analyzer.search_videos(company_info.name, company_info.ad_objective)
        
        # Initialize collection variables
        transcript_text = ""
        parameters_list = []
        ratings_dict = {}
        impact_metrics_dict = {}
        insights_list = []
        all_comments = []
        video_analysis_results = []
        
        # Analyze each video
        for video in videos_found:
            video_details = youtube_analyzer.get_video_details(video['id'])
            if not video_details:
                continue
                
            # Get transcript
            video_transcript = youtube_analyzer.get_video_transcript(video['id'])
            transcript_text += video_transcript
            
            # Get parameters and analysis
            parameters = video_analyzer.get_analysis_parameters(
                company_info.domain,
                company_info.ad_objective
            )
            parameters_list.append(parameters)
            
            # Get ratings
            current_ratings = video_analyzer.analyze_video_parameters(
                video_details,
                parameters
            )
            ratings_dict.update(current_ratings)
            
            # Calculate impact metrics
            current_impact_metrics = video_analyzer.calculate_impact_metrics(
                video_details,
                current_ratings
            )
            impact_metrics_dict.update(current_impact_metrics)
            
            # Generate report and insights
            report = video_analyzer.generate_analysis_report(
                video_details,
                current_impact_metrics
            )
            current_insights = youtube_analyzer.generate_insights(report)
            insights_list.extend(current_insights)
            
            # Get comments
            comments = youtube_analyzer.fetch_youtube_comments(video['id'])
            all_comments.extend(comments)
            
            # Store video-specific results
            video_analysis_results.append({
                "video_id": video['id'],
                "title": video_details['title'],
                "metrics": current_impact_metrics,
                "insights": current_insights
            })
        
        # Get top parameters and impact metrics
        top_5_params = sorted(ratings_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_impact_metrics = sorted(
            impact_metrics_dict.items(), 
            key=lambda x: x[1]['overall_impact'], 
            reverse=True
        )[:5]
        
        # Analyze sentiment
        sentiment_results = await video_analyzer.analyze_sentiment(all_comments)
        
        return VideoAnalysisResponse(
            competitors=competitors,
            video_analysis=video_analysis_results,
            top_parameters=top_5_params,
            top_impact_metrics=top_5_impact_metrics,
            insights=insights_list,
            sentiment_analysis=sentiment_results,
            transcript_length=len(transcript_text)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Video Analysis API"}

import os
import asyncio
from dotenv import load_dotenv
import requests
import openai
from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.preprocessing import MinMaxScaler
import asyncio
from textblob import TextBlob
from googletrans import Translator
# Load environment variables
load_dotenv()

class VideoParameterAnalyzer:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.client = OpenAI(api_key=api_key)
        self.scaler = MinMaxScaler()

    def get_analysis_parameters(self, domain: str, ad_objective: str) -> List[str]:
        """Get relevant analysis parameters based on domain and ad objective."""
        try:
            prompt = f"""Given a company in the {domain} domain with the ad objective of '{ad_objective}',
            list exactly 8 key parameters that would most influence video ad performance.
            Consider only parameters that can be extracted from video content, description, or viewer response.
            Format: Return ONLY a comma-separated list of 8 parameters, no explanations or additional text."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a marketing analytics expert. Provide concise, specific parameters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            parameters = response.choices[0].message.content.strip().split(',')
            return [param.strip() for param in parameters]
        except Exception as e:
            raise Exception(f"Error getting analysis parameters: {str(e)}")

    def analyze_video_parameters(self, video_data: Dict[str, any], parameters: List[str]) -> Dict[str, float]:
        """Analyze how well a video matches each parameter."""
        try:
            # Combine all textual content for analysis
            content = f"""
            Title: {video_data['title']}
            Description: {video_data['description']}
            Transcript: (Transcript not available)
            """  # Placeholder since transcript is not included in video_data

            # Create analysis prompt
            parameters_str = '\n'.join([f"{i+1}. {param}" for i, param in enumerate(parameters)])
            prompt = f"""Analyze this video content and rate how well it matches each parameter on a scale of 0-10.
            
            Content:
            {content}

            Parameters to rate:
            {parameters_str}

            Return ONLY numbers in order, comma-separated (e.g., '7.5,6.2,8.1,...')."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a video content analyzer. Provide numerical ratings only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            ratings = [float(x.strip()) for x in response.choices[0].message.content.strip().split(',')]
            return dict(zip(parameters, ratings))

        except Exception as e:
            raise Exception(f"Error analyzing video parameters: {str(e)}")

    def calculate_impact_metrics(self, video_data: Dict[str, any], parameter_ratings: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate the impact of each parameter on engagement metrics."""
        try:
            views = int(video_data['viewCount'])
            likes = int(video_data['likeCount'])
            comments = int(video_data['commentCount'])
            
            like_view_ratio = (likes / views) if views > 0 else 0
            comment_view_ratio = (comments / views) if views > 0 else 0
            
            impact_metrics = {}
            for param, rating in parameter_ratings.items():
                norm_rating = rating / 10.0
                
                like_impact = norm_rating * like_view_ratio
                comment_impact = norm_rating * comment_view_ratio
                
                impact_metrics[param] = {
                    'parameter_score': rating,
                    'like_ratio_impact': like_impact,
                    'comment_ratio_impact': comment_impact,
                    'overall_impact': (like_impact + comment_impact) / 2
                }
            
            return impact_metrics
            
        except Exception as e:
            raise Exception(f"Error calculating impact metrics: {str(e)}")

    def generate_analysis_report(self, video_data: Dict[str, any], impact_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate a detailed analysis report for a video."""
        try:
            metrics_str = '\n'.join([
                f"Parameter: {param}"
                f"\n  - Score: {metrics['parameter_score']:.2f}/10"
                f"\n  - Like Impact: {metrics['like_ratio_impact']:.4f}"
                f"\n  - Comment Impact: {metrics['comment_ratio_impact']:.4f}"
                f"\n  - Overall Impact: {metrics['overall_impact']:.4f}"
                for param, metrics in impact_metrics.items()
            ])
            
            prompt = f"""Analyze this video's performance based on the following metrics:

            Video Title: {video_data['title']}
            Views: {video_data['viewCount']}
            Likes: {video_data['likeCount']}
            Comments: {video_data['commentCount']}

            Parameter Impact Analysis:
            {metrics_str}

            Provide a concise analysis of:
            1. Which parameters had the strongest impact on engagement
            2. How these parameters influenced like-to-view and comment-to-view ratios
            3. Recommendations for improving engagement based on these findings"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a video performance analyst. Provide clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Error generating analysis report: {str(e)}")


    async def analyze_sentiment(self, comments):
            translator = Translator()  # Translator instance
            positive = 0
            negative = 0

            for comment in comments:
                # Await the translation coroutine
                translated = await translator.translate(comment, src='auto', dest='en')
                translated_text = translated.text

                # Create a TextBlob object for sentiment analysis
                blob = TextBlob(translated_text)

                # Get sentiment polarity
                sentiment = blob.sentiment

                # Count positive and negative sentiments
                if sentiment.polarity > 0:
                    positive += 1
                else:
                    negative += 1

            total = positive + negative
            return {
                "positive": (positive * 100) / total,
                "negative": (negative * 100) / total
            }

class CompanyInfo:
    def __init__(self, name: str, domain: str, description: str, ad_objective: str):  # Fixed initialization
        self.name = name
        self.domain = domain
        self.description = description
        self.ad_objective = ad_objective

class CompetitorAnalyzer:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.client = openai(api_key=api_key)

    def get_competitors(self, company_info: CompanyInfo) -> List[str]:
        try:
            prompt = f"""Given a company with the following details, list exactly 3 of the biggest competitors in India:
            Company Name: {company_info.name}
            Domain: {company_info.domain}
            Description: {company_info.description}
            Advertisement Objective: {company_info.ad_objective}
            
            Format your response as a simple comma-separated list of company names only."""

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Fixed model name from gpt-4o to gpt-4
                messages=[
                    {"role": "system", "content": "You are a business analyst focused on Indian market competition and advertising analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            competitors = response.choices[0].message.content.strip().split(',')
            return [comp.strip() for comp in competitors[:3]]
        except Exception as e:
            raise Exception(f"Error getting competitors: {str(e)}")

class YouTubeAnalyzer:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.client = openai(api_key=api_key)
        
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.youtube_api_key:
            raise ValueError("Missing YouTube API key")
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def fetch_youtube_comments(self,video_id):
        comments = []
        base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": os.getenv("YOUTUBE_API_KEY"),
            "maxResults": 100
        }
        while True:
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching comments for video {video_id}: {response.json()}")
                break
            data = response.json()
            for item in data.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(comment["textDisplay"])
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
            params["pageToken"] = next_page_token
        return comments

    def translate_to_english(self, text: str) -> str:
       """Translate non-English text to English using GPT-4."""
       try:
           prompt = f"""Translate the following text to English if it's not already in English. 
           If it's already in English, respond with "ALREADY_ENGLISH" exactly.
           
           Text: {text[:4000]}"""  

           response = self.client.chat.completions.create(
               model="gpt-4o",
               messages=[
                   {"role": "system", "content": "You are a professional translator."},
                   {"role": "user", "content": prompt}
               ],
               temperature=0.3
           )
           
           result = response.choices[0].message.content.strip()
           return text if result == "ALREADY_ENGLISH" else result

       except Exception as e:
           print(f"Error in translation: {str(e)}")
           return text  

    def get_video_transcript(self, video_id: str) -> str:
       """Get the transcript of a YouTube video in any available language and translate if needed."""
       try:
           transcript_list = None
            
           try:
               transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
           except Exception as e:
               try:
                   available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                   transcript_list = available_transcripts.find_transcript(['hi', 'en', 'auto']).fetch()
               except Exception as inner_e:
                   print(f"Error getting transcript list: {str(inner_e)}")
                   return ""

           if transcript_list:
               combined_transcript = ' '.join([t['text'] for t in transcript_list])
               translated_transcript = self.translate_to_english(combined_transcript)
               return translated_transcript
            
           return ""

       except Exception as e:
           print(f"Error getting transcript for video {video_id}: {str(e)}")
           return ""
        
    def check_relevance_with_gpt(self, comment: Dict[str, any], transcript: str, search_query: str) -> bool:
       """Check relevance of comments to video topic."""
       try:
           prompt = f"""Analyze if this comment is relevant to the video topic and search query.
           
           Search Query: "{search_query}"
           Video Transcript Summary: {transcript[:1000] if transcript else "No transcript available"}
           
           Comment:
           Author: {comment['author']}
           Text: {comment['text']}
           
           Consider these criteria:
           1. Does the comment discuss the ad campaign or related marketing?
           2. Is it related to the company or its competitors?
           3. Does it mention the specific campaign objective?
           4. Is it spam, generic, or off-topic?
           
           Respond with EXACTLY 'yes' if relevant or 'no' if irrelevant."""

           response = self.client.chat.completions.create(
               model="gpt-4o",
               messages=[
                   {"role": "system", "content": "You are a strict content relevance analyzer."},
                   {"role": "user", "content": prompt}
               ],
               temperature=0.3  
           )
           
           result = response.choices[0].message.content.strip().lower()
           print(f"Relevance check for comment by {comment['author']}: {result}")  
           return result == "yes"
           
       except Exception as e:
          print(f"Error in relevance check: {str(e)}")
          return False
    

    def search_videos(self, company_name: str, ad_objective: str, max_results: int = 3) -> List[Dict]:
       """Search for YouTube videos related to company and ad objective."""
       search_query = f"{company_name} {ad_objective} ad campaign"
       search_url = f"{self.base_url}/search"
       params = {
          "part": "snippet",
          "q": search_query,
          "type": "video",
          "maxResults": max_results,
          "key": self.youtube_api_key
       }
       
       response = requests.get(search_url, params=params)
       if response.status_code != 200:
          raise Exception(f"YouTube API error: {response.text}")
          
       data = response.json()
       videos = []
       for item in data.get("items", []):
          videos.append({
              "id": item["id"]["videoId"],
              "title": item["snippet"]["title"],
              "description": item["snippet"]["description"],
              "publishedAt": item["snippet"]["publishedAt"],
              "thumbnails": item["snippet"]["thumbnails"]
          })
       return videos

    def get_video_details(self, video_id: str) -> Dict:
       """Get detailed information about a specific video."""
       video_url = f"{self.base_url}/videos"
       params = {
          "part": "snippet,statistics",
          "id": video_id,
          "key": self.youtube_api_key
       }
       
       response = requests.get(video_url, params=params)
       if response.status_code != 200:
          return {}
          
       data = response.json()
       if not data.get("items"):
          return {}
          
       video_data = data["items"][0]
       return {
          "title": video_data["snippet"]["title"],
          "description": video_data["snippet"]["description"],
          "publishedAt": video_data["snippet"]["publishedAt"],
          "viewCount": int(video_data["statistics"].get("viewCount", 0)),
          "likeCount": int(video_data["statistics"].get("likeCount", 0)),
          "commentCount": int(video_data["statistics"].get("commentCount", 0))
      }

    def generate_insights(self,report_text):
            # Construct the prompt to send to GPT
            prompt = f"Please read the following report and generate a few important insights from it:\n\n{report_text}\n\nImportant Insights:"

            response = self.client.chat.completions.create(
               model="gpt-4o",
               messages=[
                   {"role": "system", "content": "You are a analyser give correct insights."},
                   {"role": "user", "content": prompt}
               ],
               temperature=0.3  
           )
            # Get the response text from OpenAI
            insights = response.choices[0].message.content.strip().split(',')
            return insights




def main():
        print("Starting analysis...")
        # Load environment variables
        load_dotenv()

        # First check if API keys are available
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: Missing OPENAI_API_KEY in .env file")
            return
        if not os.getenv('YOUTUBE_API_KEY'):
            print("Error: Missing YOUTUBE_API_KEY in .env file")
            return
        
        # Create company info instance
    
        company_info = CompanyInfo(
                name="Example Company",
                domain="Tech",
                description="A tech company specializing in AI.",
                ad_objective="Increase brand awareness"
        )
      
        
        # Initialize analyzers
        competitor_analyzer = CompetitorAnalyzer()
        youtube_analyzer = YouTubeAnalyzer()
        video_analyzer = VideoParameterAnalyzer()
        
        # Get competitors
        print("Getting competitors...")
        competitors = competitor_analyzer.get_competitors(company_info)
        print("\nCompetitors found:", competitors)
        
        # Search for videos
        # print("\nSearching for videos...")
        videos_found = youtube_analyzer.search_videos(company_info.name, company_info.ad_objective)
        # print(f"Found {len(videos_found)} videos")
        tt = ""
        params = []
        rating = {}
        rep = ""
        imp = {}
        ins = []
        comments = []
        for video in videos_found:
            # print(f"\nAnalyzing video: {video['title']}")
            
            video_details = youtube_analyzer.get_video_details(video['id'])
            if not video_details:
                print("Could not get video details, skipping...")
                continue
                
            # clear
            
            transcript_text =  youtube_analyzer.get_video_transcript(video['id'])
            tt = tt + transcript_text
            
            # Use the single video_analyzer instance
            parameters =   video_analyzer.get_analysis_parameters(
                company_info.domain,
                company_info.ad_objective
            )
            # print("\nAnalysis parameters:", parameters)
            params.append(parameters)
            ratings_dict =  video_analyzer.analyze_video_parameters(
                video_details,
                parameters
            )
            # print("\nParameter ratings:", ratings_dict)
            rating.update(ratings_dict)
            impact_metrics =  video_analyzer.calculate_impact_metrics(
                video_details,
                ratings_dict
            )
            imp.update(impact_metrics)
            # print("\nImpact metrics:", impact_metrics)
            
            report = video_analyzer.generate_analysis_report(
                video_details,
                impact_metrics
            )
            rep = rep + report
            # print("\nAnalysis Report:")
            # print(report)
            # print("\n" + "="*50)
            insights = youtube_analyzer.generate_insights(report)
            ins.append(insights)

            comment = youtube_analyzer.fetch_youtube_comments(video['id'])
            comments.append(comment)


        print(f"Transcript length: {len(transcript_text)} characters")
        top_5_params = sorted(rating.items(), key=lambda x: x[1], reverse=True)[:5]
        print(top_5_params)
        top_5_impact_metrics = sorted(imp.items(), key=lambda x: x[1]['overall_impact'], reverse=True)[:5]
        print(top_5_impact_metrics)
        print("\nGenerated Insights:")
        print(ins)
        sentiment_results = asyncio.run(video_analyzer.analyze_sentiment(comments))
        print(sentiment_results)
if __name__ == "__main__":
    (main())
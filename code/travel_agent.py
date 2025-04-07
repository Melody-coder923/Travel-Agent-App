from textwrap import dedent
from agno.agent import Agent,RunResponse
from agno.tools.serpapi import SerpApiTools
from agno.tools.duckduckgo import DuckDuckGoTools
import streamlit as st
from agno.models.groq import Groq
from agno.tools.dalle import DalleTools
from agno.utils.log import logger

from datetime import datetime
import json
from rich.pretty import pprint
from typing import List
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agno.tools.file import FileTools
from pathlib import Path

import io  # Import the io module
import base64  # Import the base64 module
import os,sys


# Set up the Streamlit app
st.title("AI Travel Planner ✈️")
st.caption("Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary on autopilot using llama")

# Get Groq API key from user
groq_api_key = "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"

# Get SerpAPI key from the user
serp_api_key = "3ee1ceb9aef090663ad0ee7ebbe82b2ecc15efb82a8eebedf78a4839a68a32c1"

#Define Class
class DailyActivity(BaseModel):
    description: str = Field(
        ..., 
        description="The description of the activity for the day "
        "and includes the day number as prefix, e.g 'Day 1: Arrive'"
    )
class Itinerary(BaseModel):
    days: List[DailyActivity] = Field(
        ...,
        description="A list of activities for each day of the trip",
    )


class Cost(BaseModel):
    total_cost: int=Field(...,
        description="The overall cost of the trip, including all expenses.",
        )
    accomodation_cost: int=Field(...,
        description="The cost of lodging during the trip, such as hotels or Airbnb rentals.",
        )
    transport_cost: int=Field(...,
        description="The cost of transportation, including flights, trains, car rentals, or public transit.",
        )
    ticket_cost: int=Field(...,
        description="The cost of tickets for attractions, tours, or events during the trip.",
        )
    food_cost: int=Field(...,
        description="The cost of meals and snacks during the trip",
        )

class Output(BaseModel):
    destination: str = Field(
        ...,
        description="The main destination of the trip."
    )
    duration: int = Field(
        ...,
        description="The number of days for the trip."
    )
    cost: Cost = Field(
        ...,
        description="The cost details of the trip."
    )

    itinerary : Itinerary = Field(
        ...,
        description="The detailed itinerary of the trip",
    )
    

    # Agent 1: Researcher (Same as v12 - provides detailed text summary)
researcher = Agent(
    name="TravelResearcher",
    role="Travel Information Gatherer",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    description=dedent("Gather travel info using web search and provide a well-structured summary of findings."),
    instructions=[
        "1. Analyze Request: Understand destination, duration, group size, budget.",
        "2. Formulate Queries: Generate 3-5 focused search queries for relevant activities, potential hotels (considering budget/group size), typical costs (per person if possible), and specific locations/addresses.",
        "3. Execute Search: Use available tools to execute web searches for each query.",
        "4. Synthesize & Format Findings: Compile a clear summary of the most relevant findings. Structure the output text like this:",
        "   ```text",
        "   **Potential Activities:**",
        "   - [Activity Name 1]: [Brief Description]. Cost: [e.g., $50/person or $100 total or Free]. Location: [Specific Address or Area].",
        "   - [Activity Name 2]: [Brief Description]. Cost: [e.g., $XX]. Location: [Specific Address or Area].",
        "   ...(List 5-10 relevant activities found)...",
        "",
        "   **Hotel Suggestions:**",
        "   - [Hotel Name 1]: Approx. [Price /Night, e.g., $150]. Address: [Hotel Address]. Note: [Brief reasoning/feature found].",
        "   ...(List 1-3 relevant hotels found)...",
        "",
        "   **General Cost Notes:**",
        "   - Daily Food Estimate: ~$X per person.",
        "   - Local Transport Estimate: ~$Y per person per day.",
        "   ```",
        "5. Return Summary: Output *only* this structured text summary. Be informative but reasonably concise. Ensure costs and locations are included for each item.",
    ],
    tools=[DuckDuckGoTools()],
    add_datetime_to_instructions=True,
)

# Agent 2: Planner (Instructions updated for integer hotel price)
planner = Agent(
    name="ItineraryPlanner",
    role="Travel Itinerary Scheduler & Strict Formatter",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    description=dedent("Create a detailed itinerary by scheduling activities from research, then strictly format the output according to the Pydantic model, ensuring clean data types and values (especially integer costs)."),
    instructions=[
        "1. Receive Input: User request (destination, days, people, budget) AND a structured text block summarizing research findings.",
        "2. Analyze & Select: Review user request and potential activities/hotels from research.",
        "3. Schedule Activities: Create a logical day-by-day itinerary for the specified duration, selecting activities from the research list. Ensure a balanced schedule.",
        "4. Determine Daily Details & Costs: For EACH scheduled day:",
        "   - Identify the main activities.",
        "   - Calculate the final estimated cost as a single **integer** value in USD for that day's scheduled activities (for the whole group). Resolve ranges/per-person costs. Store this single integer.",
        "   - Identify the most relevant specific address string.",
        "5. Select Hotel & Determine Price: Select ONE hotel from the research suggestions. Determine its estimated price per night as a single **integer** value (e.g., midpoint of range).",
        "6. Calculate Aggregate Costs: Based on the scheduled plan, the selected hotel (using the integer price from step 5), and daily cost notes: Calculate the *total* **integer** costs for `accommodation`, `transport`, `tickets`, and `food` for the entire trip. Sum these to get the overall `total_cost` **integer**.",
        "7. **CRITICAL FINAL FORMATTING STEP:** Before generating the final JSON output:",
        "   - Review ALL fields.",
        "   - Ensure ALL `Cost` fields (`total_cost`, `accommodation_cost`, etc.) contain **only integer values** based on your step 6 calculations.",
        "   - Ensure EACH `DailyActivity.estimated_cost` contains **only the single integer value** calculated in step 4. **NO ranges, '$', '/person', text.**",
        "   - Ensure EACH `DailyActivity.address` contains a clean, specific address string or relevant area name.",
        "   - Ensure `HotelSuggestion.estimated_price_per_night` contains **only the single integer value** determined in step 5. **NO ranges, '$'.**",
        "   - Verify all other string fields contain only the intended text.",
        "8. Generate Output: Structure the *final, reviewed, and correctly formatted* data strictly according to the `Output` Pydantic model, ensuring all integer fields contain only numbers.",
        "9. Suggest Image Prompts: Generate 2-3 prompts based on the scheduled activities.",
    ],
    add_datetime_to_instructions=True,
    response_model=Output,
)

    
    
# Input fields for the user's destination,the number of days, and estimated travle expense they want to travel for
destination = st.text_input("Where do you want to go?")
num_days = st.number_input("How many days do you want to travel for?", min_value=0, max_value=30)
num_persons=st.number_input("How many persons for this trip ?",min_value=0)
total_budget=st.number_input("How much is the estimated travel expense (US dollar) ?",min_value=0)
if st.button("Generate Itinerary"):
    with st.spinner("Processing..."):
        # Get the response from the assistant
        research_response = researcher.run(f"{destination} for {num_days} days for {num_persons} persons under {total_budget} US dollar", stream=False)
        st.write(research_response.content)
        planner_prompt = f"""
                **User Request:** Destination: {destination}, Duration: {num_days} days, People: {num_persons}, Budget: ~${total_budget} USD.

                **Research Findings (Potential Options):**
                ```text
                {research_response.content}
                ```
                """
        response = planner.run(planner_prompt, stream=False)
        # st.write(response.content)
        if isinstance(response.content, Output):
            markdown_content = f"""
# AI Travel Planner: {destination} Itinerary

This itinerary was generated by an AI Travel Planner.

## Itinerary:
"""
            output_obj = response.content
            itinerary_obj = output_obj.itinerary
            cost_obj = output_obj.cost
            for day in itinerary_obj.days:
                st.write(day.description)
                markdown_content += f"{day.description}\n\n"
            
            
            cost_pie_chart = [cost_obj.accomodation_cost,
                                cost_obj.transport_cost,
                                cost_obj.ticket_cost,
                                cost_obj.food_cost
                                ]
            y = np.array(cost_pie_chart)
            labels = ["Accomodation", "Transport", "Ticket", "Food"]
            fig1, ax1 = plt.subplots()
            st.divider()
            st.header("Cost Breakdowns 💰")
            markdown_content += "## Cost Breakdown:"
            colors = ["#ff69b4", "#66b3ff", "#ffff99", "#ccccff"]
            def absolute_value(val):
                a  = np.round(val/100.*y.sum(), 0)
                return f'${int(a)}'
            ax1.pie(y, labels=labels, autopct=absolute_value, startangle=90,colors=colors)
            # ax1.pie(y, labels=labels, autopct=lambda p: f'{p:.1f}US$', startangle=90,colors=colors)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)
            #Save figure to a buffer for markdown embedding
            buf = io.BytesIO()
            fig1.savefig(buf, format='png')
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            markdown_content += f"""
![Cost Breakdown Pie Chart](data:image/png;base64,{data})

"""
            markdown_content += f"**Total Cost: {cost_obj.total_cost}US$**\n"
            st.subheader(f"Total Cost: {cost_obj.total_cost}US$ ")
            
            # 保存到 Markdown 文件
            with open(f"{destination}_itinerary.md", "w", encoding="utf-8") as f:
                f.write(markdown_content)
            download_file = f"{destination}_itinerary.md"
            download_data = markdown_content
            try:
                res = os.system(f"pandoc {destination}_itinerary.md -o {destination}_itinerary.pdf")
                if res == 0:
                    download_file = f"{destination}_itinerary.pdf"
            except:
                download_file = f"destination_itinerary.md"
            
            with open(download_file,"rb") as f:
                download_data = f.read()
            st.download_button(
                label="Download Travel Plan",
                data= download_data,
                file_name= download_file,
                mime="text/pdf",
            )
            

import asyncio
import os
import pandas as pd

from analyze_reviews_jev import analyze_dataframe

reviews = pd.DataFrame({
    "free_text": [
        "The receptionist was very helpful. However the GP was rude and did not listern to my concerns, The pharmacist issued my prescription correctly",
        "I could not get an appointment for three weeks.",
    ]
})

analyzed = asyncio.run(
    analyze_dataframe(
        reviews,
        api_key=os.environ["TYPESAFE_API_KEY"],
        model="jev-latest",
        timeout=30,
        max_retries=2,
        concurrency=8,
        secondary_threshold=0.6,
    )
)

print(analyzed)

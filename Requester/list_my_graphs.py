from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R

R.initialize(ravenverse_token=os.environ.get("TOKEN"))

R.get_my_graphs()
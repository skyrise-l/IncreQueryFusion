**Data Processing and Query System**
A comprehensive data management system featuring data organization, fusion, query processing, and preprocessing capabilities.

**Project Structure**
dataSystem/
Complete data organization and management files

fusion/
Data fusion implementations including:

Baseline fusion methods

Our proposed fusion system

query/
Query processing system implementation

preprocessing/
Various data preprocessing utilities and scripts

launcher/
Main application launcher and configuration

**Data Setup**
The complete dataset is available for download at:
XXXXXX

After downloading, place all data files in the /data directory at the project root.

**Installation & Execution**
Run the system using:

bash
python -m launcher.main
**Configuration**
All execution parameters and configuration options are available in launcher/main.py. Key parameters include:

--mode: Select operation mode:

system: Run the normal system execution

baseline: Run the on-demand comparison method

batch: Run the batch fusion comparison method

--full: Execute static data fusion (use with appropriate modes)

Adjust these parameters according to your specific requirements.


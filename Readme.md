# Talk to Your Heart

**An AI-powered cardiac rehabilitation assistant transforming wearable data into actionable clinical insights and accessible patient education.**

## Overview
"Talk to Your Heart" is an end-to-end AI solution designed for patients undergoing clinical cardiac rehabilitation post-surgery. By synthesizing continuous physiological data from edge wearables (ECG & PPG) with advanced foundation models, this platform bridges the communication gap between complex physiological metrics, clinical care teams, and patients. 

The system leverages a multi-agent LLM architecture augmented by clinical knowledge bases to generate automated clinical reports (like SOAP notes and exercise stress test summaries) while providing tailored conversational interfaces for both doctors and patients.

##  Key Features
* **Dual-Interface Conversational Agents:** * *For Clinicians:* A highly technical assistant capable of deep-diving into physiological metrics, analyzing trends, and referencing treatment guides.
    * *For Patients:* An empathetic, easy-to-understand educational assistant that helps patients comprehend their health status, rehab progress, and treatment options.
* **Automated Clinical Reporting:** Automatically generates formal clinical summary reports, including SOAP notes and treadmill stress test evaluations, reducing clinician administrative burden.
* **Wearable Integration:** Ingests real-time continuous data from chest straps and wrist-worn devices via Bluetooth Low Energy (BLE) and Google Fit APIs.
* **Clinical RAG Pipeline:** Grounds LLM responses in established medical science, checking patient data against clinical literature, ACC/AHA protocols, and cardiac guidelines.

##  System Architecture

Our solution is built on a scalable, 5-stage pipeline:

### 1. Edge Data Acquisition
Patient data is captured continuously using wearable devices (chest straps and wrist monitors) and transmitted via Bluetooth Low Energy (BLE) to a local edge device.

### 2. Aggregation and Contextualization
A custom **Flutter Application** running on the patient's smartphone serves as the hub. It aggregates wearable data, integrates with Google Fit APIs, and captures manual user inputs. This layer also hosts the initial patient-facing chatbot interface.

### 3. Signal Processing and Translation
Data is streamed via a WebSocket bridge to our server-side processing environment (powered by an **NVIDIA DGX Spark Cluster**). 
* **Preprocessing:** Handled via NumPy buffers and NeuroKit2.
* **Foundation Models:** Time-series physiological data is condensed using specialized foundation models (CLEF encoder/ECG-FM) into actionable physiological feature tokens.

### 4. Cognitive Processing and RAG Integration
The core reasoning engine relies on the **Qwen2.5-72B-AWQ** model operating in a multi-agent setup. 
* **Retrieval-Augmented Generation (RAG):** Context is enriched using a **ChromaDB Vector Store** loaded with established clinical literature, ACC/AHA protocols, and cardiac guidelines. 
* The LLM synthesizes the physiological tokens, patient context payload (age, BMI, health history), and retrieved literature.

### 5. Output Dissemination and Feedback Loops
The processed insights are routed to three primary outputs:
* **Rule-based Alerts:** Immediate flags for critical physiological anomalies.
* **Clinician Interface:** Detailed dashboards for care teams, complete with auto-generated formal clinical reports (editable by the clinician).
* **Patient Interface:** A simplified conversational agent outputting directly to the patient's edge device for education and clarifying questions.

##  Tech Stack
* **Frontend:** Flutter
* **Edge APIs:** Google Fit API, Bluetooth Low Energy (BLE), WebSockets
* **Data Processing:** NumPy, NeuroKit2
* **Foundation Models:** CLEF encoder, ECG-FM
* **LLM & Reasoning:** Qwen2.5-72B-AWQ, Multi-agent reasoning framework
* **Vector Database (RAG):** ChromaDB
* **Compute:** NVIDIA DGX Spark Cluster
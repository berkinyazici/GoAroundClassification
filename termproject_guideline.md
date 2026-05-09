# BBL514E - Term Project Guidelines

## 1. General Rules

- Students must form groups of 2-3 members.
- Each group selects its own project topic parallel to their thesis topics.
- The project consists of:
  - Proposal (Due date: 08.03.2026)
  - Final Report (Due date: 10.05.2026)
  - Final Presentation (with live demo)

The project must be demonstrated live in class.

## 2. Project Requirements

Each project must satisfy both academic and system requirements.

### A. Academic Requirements

The project must:

- Address a well-defined Pattern Recognition problem
- Include mathematical formulation
- Implement at least one classificaiton method
- Include proper experimental evaluation
- Provide theoretical and empirical analysis

### B. System & Deployment Requirements

The developed system must:

#### 1. Be Dockerized

The project must run inside a Docker container.

#### 2. Provide Backend Model Serving

The trained model must be exposed via an API.

Accepted frameworks:

- FastAPI
- Flask
- Django
- Similar lightweight backend frameworks

#### 3. Provide Web Interface (Client)

A simple HTML-based interface must:

- Accept user input (file upload or feature input)
- Send request to backend
- Display prediction result
- Display confidence or probability (if available)

The interface must run from the same Docker container.

## 3. Proposal Guidelines (2-4 pages)

Must include:

1. Title
2. Team members (name, student ID) & task distribution
3. Problem definition
4. Dataset description
5. Proposed methodology (include equations when relevant)
6. Literature review (minimum 5 academic sources)
7. Evaluation plan
8. Timeline

## 4. Final Report Guidelines (6-12 pages)

Must include:

- Title (with team member names)
- Abstract

### Abstract

Purpose: Provide a concise summary of the entire study (150-250 words).

It must include:

- Brief problem statement
- Main method(s) used
- Dataset description
- Key numerical results
- Main conclusion

Do **NOT** include:

- References
- Equations
- Figures
- Citations

Use past tense.

### Introduction with Literature Review

This section should:

- Introduce the problem clearly
- Explain why it is important
- Describe its relevance to Pattern Recognition
- State the objective of the study
- Clearly outline the contributions of the Project
- Include at least 10 academic references
- Summarize related approaches
- Explain what methods have been previously used
- Identify limitations in prior work
- Position your project within existing research

### Materials and Methods

This is the technical core of the report.

It should include:

#### Dataset Description

- Dataset source
- Number of samples
- Number of features
- Number of classes
- Class balance
- Preprocessing steps

#### Mathematical Formulation

- Define the problem formally
- Define labels
- Present decision rule if applicable
- Include key equations (e.g., discriminant functions, loss function)

#### Model Description

- Describe implemented algorithms (LDA, QDA, SVM, kNN, NN, etc.)
- State assumptions (Gaussian? equal covariance?)
- Explain training procedure

This section must demonstrate theoretical understanding.

### Experimental Setup

This section describes how experiments were conducted.

Include:

- Train/test split strategy or cross-validation
- Hyperparameter tuning method
- Software environment (Python version, libraries used)
- Hardware (optional)
- Evaluation metrics:
  - Accuracy
  - Precision / Recall
  - F1-score
  - ROC-AUC (if applicable)

Make sure experiments are reproducible.

### Results

This section presents objective findings.

Must include:

- Confusion matrix
- Performance metrics
- Comparison between methods (if multiple models are used)
- Graphs (ROC curves, decision boundaries, etc.)

Do **NOT** only list numbers.

Each result must be briefly explained.

### Conclusion

This section should:

- Summarize main findings
- State which method performed best and why
- Discuss limitations
- Suggest possible improvements or future work

Keep it concise and analytical.

### References

- Use consistent citation format (APA or IEEE).
- Include only academic sources.
- All cited works must appear in the reference list.

## 5. Final Presentation

Last 2 weeks

10 minutes per group

Format:

- 5 minutes PowerPoint
- 5 minutes Live Demo

Demo must show:

- Docker container running
- Web interface working
- Real-time prediction
- System responding correctly

If the system does not run during demo, technical score will be significantly reduced.

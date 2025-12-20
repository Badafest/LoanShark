# 🦈 Loan Shark

A simple interactive browser game where the player takes the role of a "loan shark" (loan lender).

## 🎮 Game Concept

### 1. Applicant Details

Each round presents a loan applicant with features such as:

- Income
- Credit Score
- Age ...

These values are generated using controlled simulation.

### 2. Player Decision

The player acts as a lender and must select:

- Approve / Reject
- If approved:
  - Interest rate
  - Loan tenure

These choices become part of the input feature vector for the ML model.

### 3. Model Prediction

A trained classification model predicts: Will this loan default or not?

Based on the prediction:

- If the loan defaults, the player loses the loan amount but receives the property value.
- If the loan does not default, the player earns the full repayment + total interest.

## 📃 To-Do

- [ ] Train the binary classifier model
- [ ] Prepare a synthetic data generator for gameplay
- [ ] Build the backend to load the model and make predictions
- [ ] Create the browser-based game UI where users review loan applications and make decisions
- [ ] Integrate the game UI with the prediction API and implement scoring logic

# ⚙️ Requirements

- .NET 10 SDK
- Visual Studio Code with C# Dev Kit & .NET Extension Pack

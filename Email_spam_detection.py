from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

emails = [
"Congratulations! You have won ₹50,000. Click here to claim your prize.",
"Dear customer, your account has been suspended. Verify now.",
"Get a free iPhone today! Limited time offer.",
"Earn money from home without any investment.",
"Exclusive deal! Buy one get one free.",
"Win a brand new car by entering this contest.",
"Click here to receive your lottery winnings.",
"Urgent! Your bank account needs verification.",
"Claim your cashback reward now.",
"Limited offer! Get 80% discount today.",
"Free vacation package waiting for you.",
"You have been selected for a special reward.",
"Act now and get a free gift card.",
"Lowest loan rates available. Apply now.",
"Your payment failed. Update details immediately.",
"Congratulations! You are our lucky winner.",
"Earn ₹10,000 daily from home.",
"Free membership for the first 100 users.",
"Special promotion just for you.",
"Click now to unlock premium features.",
"Meeting is scheduled for tomorrow at 10 AM.",
"Please find the attached project report.",
"Happy Birthday! Have a great day.",
"Can we discuss the assignment after class?",
"Your package will be delivered tomorrow.",
"Thank you for attending the meeting today.",
"The exam timetable has been uploaded.",
"Let's have lunch together this weekend.",
"Project submission deadline is Friday.",
"Your interview is scheduled for Monday.",
"Please review the document and share feedback.",
"Team meeting has been postponed.",
"The class will start at 9 AM.",
"Your order has been shipped successfully.",
"Reminder: Doctor appointment tomorrow.",
"Thanks for your help on the project.",
"Please send me the presentation slides.",
"The workshop registration is confirmed.",
"Your library books are due next week.",
"See you at the conference tomorrow.",
"Win cash prizes instantly by clicking here.",
"Get rich quickly with this secret method.",
"Exclusive investment opportunity available now.",
"Claim your free Netflix subscription.",
"Your account has won a special bonus.",
"Click here to verify your identity.",
"Limited stock available. Buy now.",
"You are pre-approved for a loan.",
"Get free cryptocurrency today.",
"Special reward waiting for confirmation.",
"Please join the Zoom meeting at 3 PM.",
"The assignment marks have been published.",
"Your train ticket has been confirmed.",
"Thank you for your purchase.",
"Please update the spreadsheet.",
"The event starts at 6 PM.",
"Your password was changed successfully.",
"The seminar has been rescheduled.",
"Please call me when you are free.",
"Let's discuss the project requirements.",
"Your package is on hold. Confirm your address information immediately to avoid return to sender.",
"Delivery failed due to incorrect address details. Update your information within 24 hours.",
"We attempted to deliver your parcel today. Verify your shipping address to schedule redelivery.",
"Your shipment is waiting for confirmation. Please update your delivery details now.",
"Package delivery could not be completed. Confirm your information to avoid cancellation.",
"Important: Your parcel has been delayed. Verify your address information immediately.",
"Your order cannot be shipped until address verification is completed.",
"Final notice: Delivery attempt failed. Update your shipping details within 48 hours.",
"Your package is being held at our distribution center. Confirm your address now.",
"Action required: Verify delivery information to receive your package.",
"Security alert: We detected unusual activity on your account. Verify your account immediately.",
"Your account will be suspended unless verification is completed today.",
"Confirm your banking information to avoid service interruption.",
"We noticed suspicious login activity. Verify your account credentials now.",
"Account review required. Complete verification to maintain access.",
"You have been selected for a cashback reward. Confirm your account details now.",
"Exclusive reward waiting for you. Verify your information to claim benefits.",
"Congratulations! Your account qualifies for a special loyalty reward.",
"Claim your reward credits today by confirming your account information.",
"Limited-time customer benefit available. Verify eligibility immediately.",
"Subject: Online Banking Notice. Your account security settings require immediate review. Please verify your details to continue uninterrupted banking services.",
"Subject: Delivery Exception. We were unable to complete delivery of your parcel. Confirm your address information to schedule a new delivery attempt.",
"Subject: Membership Reward. You have been selected to receive a complimentary membership upgrade. Confirm eligibility today.",
"Subject: Security Verification Required. Suspicious activity has been detected on your account. Verify your identity immediately.",
"Subject: Prize Redemption Notice. A reward has been reserved for your account. Complete verification to claim it.",
"Subject: Employment Opportunity. Earn extra income from home with flexible working hours and no experience required.",
"Subject: Subscription Renewal Alert. Your service will expire soon. Update billing information to avoid interruption.",
"Subject: Customer Loyalty Program. Exclusive rewards are available for selected members. Confirm participation now.",
"Subject: Urgent Account Update. Failure to review your account details may result in temporary restrictions.",
"Subject: Investment Opportunity. Limited-time access to a high-return investment program is available now.",
"Subject: Faculty Meeting Reminder. The department meeting will be held on Thursday at 11:00 AM.",
"Subject: Assignment Feedback. Your submitted assignment has been reviewed and feedback is now available.",
"Subject: Office Holiday Notice. The office will remain closed on Monday due to a public holiday.",
"Subject: Conference Registration Confirmed. Your registration has been successfully processed.",
"Subject: Training Session Schedule. The next training session will begin at 9:30 AM tomorrow.",
"Subject: Project Milestone Update. The development team has completed the first project milestone.",
"Subject: Examination Results Available. Results have been uploaded to the student portal.",
"Subject: Team Lunch Invitation. You are invited to join the team lunch this Friday afternoon.",
"Subject: Monthly Performance Review. Please attend your review meeting next week.",
"Subject: Library Membership Renewal. Your library membership has been renewed successfully.",
"Subject: Account Verification Needed. Please confirm your information to prevent service disruption.",
"Subject: Shipping Update. Your package is awaiting confirmation before dispatch.",
"Subject: Scholarship Interview Schedule. Your interview has been scheduled for Wednesday.",
"Subject: Customer Reward Notice. Reward points have been credited to your account.",
"Subject: Workshop Participation Certificate. Certificates will be distributed next week.",
"Subject: Password Reset Request. A request to reset your password was received.",
"Subject: Internship Application Update. Your application is currently under review.",
"Subject: New Device Login Alert. A login from a new device was detected.",
"Subject: Course Completion Notice. Congratulations on completing the training course.",
"Subject: Premium Benefits Available. Upgrade today to unlock additional benefits.",
"Subject: Payment Confirmation. Your recent payment has been processed successfully.",
"Subject: Urgent Security Alert. Verify your account ownership immediately.",
"Subject: Seminar Invitation. You are invited to attend the upcoming seminar.",
"Subject: Cashback Offer. Claim your cashback reward before it expires.",
"Subject: Attendance Record Update. Your attendance report has been updated.",
"Subject: Special Discount Available. Enjoy exclusive discounts on selected products.",
"Subject: Parent Teacher Meeting. The meeting is scheduled for next Saturday.",
"Subject: Reward Eligibility Notice. Confirm your details to receive benefits.",
"Subject: Staff Training Program. Attendance is mandatory for all employees.",
"Subject: Free Trial Access. Activate your complimentary access today.",
"During a recent security audit, we identified unusual activity associated with your account. Review and confirm your account information to avoid temporary restrictions."

]

labels = [
1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,
1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,
1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,
1,1,0,0,0,0,0,1,0,1,
0,1,0,1,0,1,0,1,0,1,
1,1,0,1,0,1,0,0,0,1,
0,1,0,1,0,1,0,1,0,1,
1
]


# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42,stratify=labels
)
# Train Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make Predictions
nb_pred = nb_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Evaluation
print("accuracy",accuracy_score(y_test,nb_pred))
print("accuracy ",accuracy_score(y_test,svm_pred))
print("accuracy ",accuracy_score(y_test,lr_pred))


# User Input
while True:
    new_email = input("\nEnter Email (exit to quit): ")

    if new_email.lower() == "exit":
        break

    new_email_vector = vectorizer.transform([new_email])
    prediction = nb_model.predict(new_email_vector)
    prediction1 = svm_model.predict(new_email_vector)
    prediction2 = lr_model.predict(new_email_vector)

    print("\nNaive Bayes:",
      "Spam⚠️☠️ 🚨" if prediction[0] == 1 else "Not Spam ✅\n")

    print("\nLinear SVC:",
      "Spam ⚠️☠️🚨" if prediction1[0] == 1 else "Not Spam ✅\n")

    print("\nLogistic Regression:",
      "Spam ⚠️☠️🚨" if prediction2[0] == 1 else "Not Spam ✅\n")
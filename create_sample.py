import pandas as pd
df = pd.DataFrame({
  'resume_text': ['Experienced software engineer John with Python and machine learning skills he excels at',
    'Marketing specialist Mary with strong communication she led teams and her projects succeeded',
    'Data analyst with SQL and statistics background',
    'Project manager Michael with PMP certification he manages large teams effectively',
    'Junior developer Sarah with React and JavaScript she builds great interfaces',
    'Senior engineer Robert skilled in cloud architecture he mentors junior developers',
    'HR coordinator Lisa with recruiting experience she focuses on talent acquisition',
    'Business analyst James proficient in Excel and analytics he drives insights',
    'Designer Emily with UI/UX skills she creates beautiful user experiences',
    'Consultant David with strategy background he advises executive teams'],
  'label': [1,1,0,1,0,1,0,1,1,1],
  'group': ['male','female','male','male','female','male','female','male','female','male']
})
df.to_csv('sample_resumes.csv', index=False)
print('done')

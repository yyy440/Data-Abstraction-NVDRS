from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# zero shot prompt formats
first_prompt = """Answer the following multiple choice questions by abstracting from the given text.


Question: Does the text specifically state that V was in a depressed mood?

0: Yes
1: No


Question: Does the text specifically state if V was currently in treatment for a mental health or substance abuse problem?

0: Yes
1: No


Question: Does the text specifically state if V ever treated for a mental health or substance abuse problem?

0: no
1: yes


Question: Does the text specifically state if V attempted suicide before the current attempt?

0: no
1: yes

{format_instructions}

{text}

Answer: 
"""

second_prompt = """Answer the following multiple choice questions by abstracting from the given text.

Question: Did V think about suicide before committing suicide?

0: no
1: yes


Question: Did V suffer from a substance abuse problem?

0: no
1: yes

Question: Did V have a mental health condition?

0: no
1: yes


Question: Was V diagnosed with anxiety?

0: no
1: yes


{format_instructions}

{text}

Answer: 
"""

third_prompt = """Answer the following multiple choice questions by abstracting from the given text.

Question: Was V diagnosed with Depression Dysthymia?

0: no
1: yes


Question: Was V diagnosed with bipolar?

0: no
1: yes 


Question: Was V diagnosed with Adhd?

0: no
1: yes


Question: Did V have problems with their intimate partner?

0: no
1: yes


{format_instructions}

{text}

Answer: 
"""

fourth_prompt = """Answer the following multiple choice questions by abstracting from the given text.

Question: Was V having relationship problems with their friends or family?

0: no
1: yes


Question: Did an argument contribute to V's suicide?

0: no
1: yes 


Question: Did V have problems at school?

0: no
1: yes


Question: Did V have criminal legal problems? 

0: no
1: yes


{format_instructions}

{text}

Answer: 
"""

fifth_prompt = """Answer the following multiple choice questions by abstracting from the given text.

Question: Did V leave a suicide note?

0: no
1: yes


Question: Did V tell someone of their plan to commit suicide within a month of their suicide?
    
0: no
1: yes


Question: Did V tell their initimate partner about their suicidal intent?

0: no
1: yes


Question: Did V disclose their suicidal intent to family?

0: no
1: yes


{format_instructions}

{text}

Answer: 
"""


sixth_prompt = """Answer the following multiple choice questions by abstracting from the given text.

Question: Did V tell a friend of their plan for suicide?

0: no
1: yes

Question: What location did V commit suicide?

1: House, apartment
2: Motor vehicle (excluding school bus and public transportation)
3: Natural area (e.g., field, river, beaches, woods)
4: Park, playground, public use area
5: Street/road, sidewalk, alley
6: Other


Question: What method did V use to commit suicide?

1: Blunt instrument
2: Drowning
3: Fall
4: Fire or burns
5: Firearm
6: Hanging, strangulation, suffocation
7: Motor vehicle including buses, motorcycles
8: Other transport vehicle, eg, trains, planes, boats
9: Poisoning
10: Sharp instrument
11: Other (e.g. taser, electrocution, nail gun)
12: Unknown


{format_instructions}

{text}

Answer: 
"""

# For few shot prompt
example_template = """User:\n{Text}\nAI: {Answer}"""
example_prompt = PromptTemplate.from_template(template=example_template)
suffix = """{format_instructions}\nUser:\n{input}\nAI: """
prefix = """Answer the following multiple choice questions by abstracting from the given text.


Question: Does the text specifically state that V was in a depressed mood?

0: Yes
1: No


Question: Does the text specifically state if V was currently in treatment for a mental health or substance abuse problem?

0: Yes
1: No


Question: Does the text specifically state if V ever treated for a mental health or substance abuse problem?

0: no
1: yes


Question: Does the text specifically if V attempted suicide before the current attempt?

0: no
1: yes
"""

difficult_prefix = """Answer the following multiple choice questions by abstracting from the given text.


Question: Did V have problems with their family which contributed to their suicide?

0: Yes
1: No


Question: Did V have criminal legal problems that contributed to their suicide?

0: Yes
1: No


Question: Did V tell an intimate partner of their intent to commit suicide?

0: no
1: yes


Question: Did V tell a family member of their intent to commit suicide?

0: no
1: yes


Question: Did V tell a friend of their intent to commit suicide?

0: no
1: yes
"""

all_prefix = """Answer the following multiple choice questions by abstracting from the given text.


Question: Does the text state if V in a depressed mood?

0: Yes
1: No


Question: Does the text state if V was currently in treatment for a mental health or substance abuse problem?

0: Yes
1: No


Question: Does the text state if V ever treated for a mental health or substance abuse problem?

0: no
1: yes


Question: Does the text state if V attempted suicide before the current attempt?

0: no
1: yes


Question: Did V think about suicide before committing suicide?

0: no
1: yes


Question: Did V have a substance abuse problem?

0: no
1: yes


Question: Did V have a mental health condition?

0: no
1: yes


Question: Was V diagnosed with anxiety?

0: no
1: yes


Question: Was V diagnosed with Depression Dysthymia?

0: no
1: yes


Question: Was V diagnosed with bipolar?

0: no
1: yes 


Question: Was V diagnosed with Adhd?

0: no
1: yes


Question: Did V have problems with their intimate partner?

0: no
1: yes


Question: Was V having relationship problems with their friends or family?

0: no
1: yes


Question: Did an argument contribute to V's suicide?

0: no
1: yes 


Question: Did V have problems at school?

0: no
1: yes


Question: Did V have criminal legal problems? 

0: no
1: yes


Question: Did V leave a suicide note?

0: no
1: yes


Question: Did V tell someone of their plan to commit suicide within a month of their suicide?
    
0: no
1: yes


Question: Did V tell their initimate partner of their suicidal intent?

0: no
1: yes


Question: Did V disclose their suicidal intent to family?

0: no
1: yes


Question: Did V tell a friend of their plan for suicide?

0: no
1: yes

"""
# Question: What location did V commit suicide?

# 1: House, apartment
# 2: Motor vehicle (excluding school bus and public transportation)
# 3: Natural area (e.g., field, river, beaches, woods)
# 4: Park, playground, public use area
# 5: Street/road, sidewalk, alley
# 6: Other


# Question: What method did V use to commit suicide?

# 1: Blunt instrument
# 2: Drowning
# 3: Fall
# 4: Fire or burns
# 5: Firearm
# 6: Hanging, strangulation, suffocation
# 7: Motor vehicle including buses, motorcycles
# 8: Other transport vehicle, eg, trains, planes, boats
# 9: Poisoning
# 10: Sharp instrument
# 11: Other (e.g. taser, electrocution, nail gun)
# 12: Unknown





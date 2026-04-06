import sys
sys.path.insert(0, "src")

from dialogue_manager import DialogueManager

dm = DialogueManager()

question = """What is the purpose of life and death" , these are one hell of a

question my friend that detaches you from everything. Either stay away

from it my friend or let it consume you in its entirety and the beauty

of life will unfold within you and may god be with you """
print(f"USER: {question}\n")
response = dm.turn(question)
print(f"AI: {response}")
print(f"\n--- {len(dm.state.a_commitments)} commitments, {len(dm.state.aporic_questions)} aporic questions ---")

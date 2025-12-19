from typing import Dict, Any

from agent.derive_flags import derive_flags
from agent.eligibility import find_eligible_schemes
from rag.retriever import retrieve_scheme_info
from data.scheme import schemes


# Conversation state
agent_state: Dict[str, Any] = {
    "current_state": "START",
    "user_profile": {}
}

# Eligibility questions
eligibility_questions = [
    ("age", "మీ వయస్సు ఎంత?"),
    ("gender", "మీ లింగం ఏమిటి? (male / female)"),
    ("marital_status", "మీ వివాహ స్థితి ఏమిటి? (married / unmarried / widow / divorced)"),
    ("occupation", "మీ వృత్తి ఏమిటి? (farmer / student / employee / self-employed / unemployed)"),
    ("income", "మీ వార్షిక ఆదాయం ఎంత?"),
    ("bpl", "మీరు BPL కుటుంబానికి చెందినవారా? (yes / no)")
]


def agent_loop() -> None:
    """Main agent state machine loop."""

    print("👋 నమస్కారం! ప్రభుత్వ పథకాల సహాయకుడికి స్వాగతం.\n")

    # -------- ASK ELIGIBILITY --------
    agent_state["current_state"] = "ASK_ELIGIBILITY"

    for field, question in eligibility_questions:
        while True:
            answer = input(question + " ").strip()

            try:
                # yes / no normalization
                if answer.lower() in ["yes", "no"]:
                    value = answer.lower() == "yes"

                # numeric fields
                elif field in ["age", "income"]:
                    value = int(answer)

                # categorical validations
                elif field == "gender":
                    if answer not in ["male", "female"]:
                        raise ValueError
                    value = answer

                elif field == "marital_status":
                    if answer not in ["married", "unmarried", "widow", "divorced"]:
                        raise ValueError
                    value = answer

                elif field == "occupation":
                    if answer not in [
                        "farmer", "student", "employee", "self-employed", "unemployed"
                    ]:
                        raise ValueError
                    value = answer

                else:
                    value = answer

                agent_state["user_profile"][field] = value
                break

            except ValueError:
                print("❌ సరైన సమాచారం ఇవ్వలేదు. దయచేసి మళ్లీ ప్రయత్నించండి.")

    # -------- DERIVE FLAGS --------
    agent_state["user_profile"] = derive_flags(agent_state["user_profile"])

    # -------- ELIGIBILITY CHECK --------
    agent_state["current_state"] = "CHECK_ELIGIBILITY"
    eligible = find_eligible_schemes(agent_state["user_profile"], schemes)

    if not eligible:
        print("\n❌ మీకు ప్రస్తుతం అర్హత ఉన్న పథకాలు లేవు.")
        return

    # -------- MAIN SCHEME LOOP --------
    while True:
        agent_state["current_state"] = "SHOW_SCHEMES"
        print("\n✅ మీకు అర్హత ఉన్న పథకాలు:\n")

        for i, s in enumerate(eligible, 1):
            print(f"{i}. {s['scheme_name']} – {s['benefits']}")

        choice = input(
            "\nఏ పథకం వివరాలు కావాలి?\n"
            "పథకం నంబర్ ఇవ్వండి లేదా 'exit' టైప్ చేయండి: "
        ).strip()

        if choice.lower() == "exit":
            print("\n🙏 ధన్యవాదాలు!")
            return

        try:
            selected_scheme = eligible[int(choice) - 1]
        except (ValueError, IndexError):
            print("❌ సరైన ఎంపిక ఇవ్వలేదు. మళ్లీ ప్రయత్నించండి.")
            continue

        # -------- EXPLAIN SCHEME (RAG) --------
        agent_state["current_state"] = "EXPLAIN_SCHEME"

        print(f"\n📌 పథకం: {selected_scheme['scheme_name']}")

        scheme_info = retrieve_scheme_info(
            "ఈ పథకం వివరాలు చెప్పండి",
            selected_scheme["scheme_id"]
        )

        print("\nℹ️ పథకం వివరాలు:\n")
        print(scheme_info)

        # -------- APPLY INFO --------
        want_apply = input("\nఈ పథకానికి దరఖాస్తు విధానం తెలుసుకోవాలా? (yes / no) ").strip()

        if want_apply.lower() == "yes":
            agent_state["current_state"] = "APPLY_GUIDANCE"

            apply_info = retrieve_scheme_info(
                "ఈ పథకానికి దరఖాస్తు ఎలా చేయాలి?",
                selected_scheme["scheme_id"]
            )

            print("\n📝 దరఖాస్తు చేసే విధానం:\n")
            print(apply_info)

        # -------- EVALUATION --------
        again = input("\nఇంకొక పథకం వివరాలు కావాలా? (yes / no) ").strip()

        if again.lower() != "yes":
            print("\n✅ మీకు సహాయం చేయగలిగినందుకు సంతోషం!")
            return
if __name__ == "__main__":
    agent_loop()
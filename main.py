from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the first and only trillionaire in terms of US dollars in mid June 2026; however as of June 21, 2026, Forbes estimates his net worth to be US$951 billion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service; he implemented significant changes and rebranded it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion.

Musk is a supporter of global far-right politics, figures, and political parties. He was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated in January 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump.

Musk's political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including spreading COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, white nationalist, racist, and transphobic comments. His acquisition of Twitter was controversial because, following his pledge to decrease censorship, there was an increase in hate speech and misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE and its cuts to the US Agency for International Development (USAID).

Early life and education
See also: Musk family
Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[2][3] He is of British and Pennsylvania Dutch ancestry.[4][5] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[6][7][8] Musk therefore holds both South African and Canadian citizenship from birth.[9] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[10][11][12][13]

His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Musk was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the Technocracy movement[14][15] who moved to South Africa in 1950.[16] Haldeman's anti-government, anti-democratic and conspiracist views, which included the promotion of far-right antisemitic conspiracy theories,[17][18] "fanatical" support of apartheid,[18] and according to Errol, support of Nazism,[16] have been suggested as an influence on Musk.[19][20][21][22] During his childhood, Musk was told stories by his grandmother of Haldeman's travels and exploits, and he has suggested that all of Haldeman's descendants have his "desire for adventure, exploration – doing crazy things".[23].
    """


    summary_template = """
    Summarize the following information about {information} to create:
    1. A short summary of the information
    2. Two interesting facts about the them
    """

    summary_prompt = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #llm = ChatOllama(model="gemma3:4b", temperature=0)
    summary_chain = summary_prompt | llm

    response = summary_chain.invoke(input={"information": information})

    print(response.content)

if __name__ == "__main__":
    main()

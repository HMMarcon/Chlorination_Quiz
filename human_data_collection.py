import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from itertools import chain

from rdkit.Chem import rdChemReactions, rdCoordGen, rdDepictor




def possible_prods(starting_material):
    """
    Input: RDKit Mol-object of starting material

    Output: List of SMILES of each mono-chlorination combination in aromatic system.
            If no aromatic system is recognized, it returns from all CH bonds that might become C-Cl
    """

    potential_products_Ar = []
    molecule = Chem.AddHs(starting_material)

    for atom in molecule.GetAtoms():
        pot_product = molecule

        if atom.GetSymbol() == "H":
            for neighbour in atom.GetNeighbors():
                if neighbour.GetIsAromatic() and neighbour.GetAtomicNum() == 6:
                    atom.SetAtomicNum(17)
                    potential_products_Ar.append(Chem.MolToInchi(pot_product))
                    atom.SetAtomicNum(1)

    if len(potential_products_Ar) == 0:  # In case RDKit does not manage to find an aromatic ring
        for atom in molecule.GetAtoms():
            pot_product = molecule

            if atom.GetSymbol() == "H":
                for neighbour in atom.GetNeighbors():
                    if neighbour.GetAtomicNum() == 6:
                        atom.SetAtomicNum(17)
                        potential_products_Ar.append(Chem.MolToInchi(pot_product))
                        atom.SetAtomicNum(1)

    potential_products_Ar = set(potential_products_Ar)
    potential_products = [Chem.MolFromInchi(molecule) for molecule in potential_products_Ar]
    potential_products = [Chem.MolToSmiles(molecule) for molecule in potential_products]
    return potential_products
def draw_molecules_with_names(smiles_list):
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    names_list = [f'Product {i+1}' for i in range(len(smiles_list))]
    img = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(200, 200),
                               legends=names_list, useSVG = False, returnPNG = False)
    return img
def correct_pick(selected, correct):
    acc = []
    for i in range(len(selected)):
        selected_inchi = Chem.MolToInchi(Chem.MolFromSmiles(selected[i]))
        correct_inchi = Chem.MolToInchi(Chem.MolFromSmiles(correct[i]))
        if  selected_inchi == correct_inchi:
            acc.append(1)
        else:
            acc.append(0)
    return acc
def get_new_atom(substrate: Chem.Mol, product: Chem.Mol):
  """
  Assumes implicit H atoms for faster performance

  arguments:
  - substrate: RDKit mol for substrate
  - product: RDKit mol for product
  returns:
  - list containing the index of NEW Cl atom
  """
  substructure = product.GetSubstructMatch(substrate)
  chlorines = []

  for atom in product.GetAtoms():
    if atom.GetIdx() not in substructure:
      chlorines.append(atom.GetIdx())

  return chlorines
def get_drawer_options():
  '''auxiliary function, returns options for drawer
  more options at: https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html#rdkit.Chem.Draw.rdMolDraw2D.MolDrawOptions
  '''
  drawer = Draw.MolDraw2DCairo(-1,-1)
  opts = drawer.drawOptions()
  opts.baseFontSize=0.8
  opts.addAtomIndices=False
  opts.highlightRadius = 0.9
  opts.highlightBondWidthMultiplier = 24
  opts.legendFontSize = 36
  #color = (0.88, 0.88, 0.88)
  #color = tuple(i/255 for i in (179,235,206))
  color = tuple(i/255 for i in (185,242,213))
  opts.setHighlightColour(color)
  return opts
def nicer_display(smiles:str):
    """
    Arguments:
    - smiles: SMILES for the molecule.
    """
    # create mol from supplied SMILES
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)

    # instantiate RDKit chemical reaction
    chlorinate_smarts = '[cH1:1]>>[cH0:1]-Cl'
    chlorinate = rdChemReactions.ReactionFromSmarts(chlorinate_smarts)

    # generate all possible products, return as list
    products_rdkit = chlorinate.RunReactants([mol])
    products = list(chain.from_iterable(products_rdkit))
    #for mol in products:
    #    Chem.SanitizeMol(mol)
    temp_1 = [Chem.SanitizeMol(mol) for mol in products]

    products_smiles = [Chem.MolToSmiles(mol) for mol in products]

    # align products with the original
    #for p in products:
    temp_2 = [rdDepictor.GenerateDepictionMatching2DStructure(p,mol) for p in products]
    #[rdDepictor.GenerateDepictionMatching2DStructure(p,mol) for p in products]
    drawer_opts = get_drawer_options()

    legends = [f'Product {i}' for i, p in enumerate(products, start=1)]

    highlight_atoms = [get_new_atom(mol, product) for product in products]

    img = Draw.MolsToGridImage(products, legends=legends, highlightAtomLists=highlight_atoms, drawOptions=drawer_opts,
                               useSVG = False, returnPNG = False, molsPerRow=3, subImgSize=(500, 500))
    return img, legends, products_smiles

def draw_results (df):
    """
    Input: list of SMILES for the reactants, list of correct products, list of selected products
    Output: Drawn molecules with correct and selected products
    """
    mols = []
    legends = []
    highlight_atoms = []
    # Specify only the columns with SMILES strings
    smiles_columns = ['selected_product', 'ai_prediction', 'correct_product']
    df_raw = pd.DataFrame(df)
    # Convert SMILES to RDKit Molecules in the specified columns
    for col in smiles_columns:
        df_raw[col] = df_raw[col].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
        #df_raw[col] = df_raw[col].apply(lambda x: Chem.SanitizeMol(x) if x else None)


    for i, row in df_raw.iterrows():
        for col in smiles_columns:
            mols.append(row[col])

            highlight_atoms.append(get_new_atom(Chem.MolFromSmiles(row['smiles_sm']), row[col]))
        legends.append([f'Your Pick {i+1}',f'AI Prediction {i+1}', f'Correct Answer {i+1}'])
    legends = list(chain.from_iterable(legends))



    temp1 = [Chem.SanitizeMol(mol) for mol in mols]

    drawer_opts = get_drawer_options()
    # Draw grid image

    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500), drawOptions=drawer_opts,
                               highlightAtomLists=highlight_atoms, legends = legends,
                               useSVG= False, returnPNG = False)

    return img

@st.cache_data
def read_data():
    rxn_df = db.read(worksheet="Raw_data", usecols=["Reactant", "Product", "AI"], nrows=1782)

    return rxn_df


n_samples = 5
time = []
random_baseline = 1
background_list = []
selections_list = []
experience_list = []
correct_list = []
selection = []
random_examples = []
finished = False
st.title("Organic Chemistry Quiz")

st.write("Test your knowledge of aromatic substitutions by selecting the correct product for each reaction. "
         "You will be asked to select the correct product for 5 reactions. "
         "\n \n Your submissions will be recorded for analysis.")


with st.expander("How to use this page"):
    st.markdown("Using your chemical expertiese we want to see how challenging it is to predict aromatic substitution reactions.")
    st.markdown("- For each molecule (right), you will be presented with its mono-chlorination products "
                "(below).")
    st.markdown("- From the drop-down menu in the end, select which is the major product")
    st.markdown("- You will see 5 reactions, then the answers from literature and AI-prediction")
    st.markdown("- You can play again by clicking the 'reset' button")
    st.image("Schema-Chlorination-Web-Quiz.png")
    
col1, col2 = st.columns([0.5, 0.5])
with col1:
    background = st.selectbox("What is your background?", ["Chemist", "Chemical Engineer", "Other"])
    experience = st.selectbox("What is your experience with organic chemistry?", ["Undegrad/Masters", "PhD",
                                                                                   "Academic (Post-doc, Professor)",
                                                                                   "Industry", "Other"])
    age_experience = st.number_input("How many years of experience do you have with organic chemistry?", min_value=0,
                                     max_value=100, value=1, step=1)

if st.checkbox("I have added my background information."):
    try:
        db = st.connection("gsheets", type=GSheetsConnection)

        #rxn_df = pd.read_csv("input_file.csv", header=0)
        rxn_df = read_data()
        responses_df = db.read(worksheet = "Responses", usecols=["time", "background", "experience", "age_experience",
                                                                 "smiles_sm", "smiles_selected", "correct",
                                                                 "ai_prediction", "correct_product", "selected_product"])
    except:
        st.error(f"Couldn't load the data. \n\n Please, contact: hmm59@cam.ac.uk")
else:
    st.stop()

# Initialize session state variables
if 'current_iteration' not in st.session_state:
    st.session_state['current_iteration'] = 0
    st.session_state['selections_list'] = []
    st.session_state['background_list'] = []
    st.session_state['experience_list'] = []
    st.session_state['time'] = []
if 'random_examples' not in st.session_state:
    st.session_state['random_examples'] = rxn_df.sample(n=n_samples)

if 'age_experience' not in st.session_state:
    st.session_state['age_experience'] = []

if 'ai_prediction' not in st.session_state:
    st.session_state['ai_prediction'] = []

random_examples = st.session_state['random_examples']
correct_list = random_examples["Product"].tolist()
#st.image(draw_molecules_with_names(random_examples["Reactant"]))
#st.write(len(random_examples))
# Process one reaction per iteration
if st.session_state['current_iteration'] < n_samples:
    row = random_examples.iloc[st.session_state['current_iteration']]
    # Display the reactant
    with st.form(key=f"form_{st.session_state['current_iteration']}"):
        st.progress((st.session_state['current_iteration'] + 1) / n_samples)
        #st.write(st.session_state['current_iteration'] + 1, ".")

        col2.image(Draw.MolToImage(Chem.MolFromSmiles(row.Reactant),size=(350, 260),fitImage=True))

        #products_smiles = possible_prods(Chem.MolFromSmiles(row.Reactant))

        #st.image(draw_molecules_with_names(products_smiles))

        grid_image, options, products_smiles = nicer_display(row.Reactant)
        st.image(grid_image)
        random_baseline += 1/len(options)

        #options = [f'Product {j + 1}' for j in range(len(products_smiles))]

        selected_prod = st.selectbox(f"Select product for reaction {st.session_state['current_iteration'] + 1}:",
                                     options)
        selected_prod_smiles = products_smiles[options.index(selected_prod)]

        ai_prediction = row.AI

        if st.session_state['current_iteration'] < n_samples - 1:  # last one
            action = "Next"
        else:
            action = "Submit"

        submit_button =st.form_submit_button(label=action)
        if submit_button:
            # Append selections to the session state
            st.session_state['selections_list'].append(selected_prod_smiles)
            st.session_state['background_list'].append(background)
            st.session_state['experience_list'].append(experience)
            st.session_state['time'].append(datetime.now())
            st.session_state['age_experience'].append(age_experience)
            st.session_state['ai_prediction'].append(ai_prediction)



            # Move to next iteration
            st.session_state['current_iteration'] += 1
            st.rerun()

# Check if all iterations are done
with col2:

    if st.session_state['current_iteration'] >= len(random_examples):


        selections_list = st.session_state['selections_list']
        background_list = st.session_state['background_list']
        experience_list = st.session_state['experience_list']
        ai_prediction_list = st.session_state['ai_prediction']
        time = st.session_state['time']
        #st.write(st.session_state['selections_list'])
        correct_list = correct_pick(selections_list, random_examples["Product"].to_list())
        ai_correct_list = correct_pick(ai_prediction_list, random_examples["Product"].to_list())

        # Process the collected data here

        # Save the results
        if len(selections_list) == len(random_examples) and len(time) == len(random_examples) and len(correct_list) == len(random_examples) and len(background_list) == len(random_examples):
            # Combine new data with sampled rows
            new_data = pd.DataFrame()
            new_data['time'] = time
            new_data['background'] = background_list
            new_data['experience'] = experience_list
            new_data['age_experience'] = st.session_state['age_experience']
            new_data['smiles_sm'] = random_examples['Reactant'].to_list()
            new_data['selected_product'] = selections_list
            new_data['ai_prediction'] = ai_prediction_list
            new_data['correct_product'] = random_examples['Product'].to_list()


            new_data['correct'] = correct_list #function here to compare correct and selected

            performance = round(sum(correct_list) / len(correct_list) * 100, 0)
            ai_performance = round(sum(ai_correct_list) / len(ai_correct_list) * 100, 0)
            random = round(random_baseline * 100/ len(correct_list), 0)
            st.markdown(f"All reactions processed. You got **:green[{round(performance)}% correct]**  \n\n"
                        f"AI performance = **:orange[{round(ai_performance)}%]**  \n\n"
                        f"Random-pick = **:red[{round(random)}%]**  \n\n")

            if performance > ai_performance and performance > random:
                st.markdown(f"Well done! AI won't be replacing you soon.")
            elif performance > (ai_performance+random)/2 and performance > random:
                st.markdown(f"Great result! Keep practicing.")
            elif performance > random:
                st.markdown(f"Better than random guess!")
            else:
                st.markdown(f"You might want to revise the topic. ")


            #existing_data = pd.read_csv("human_outputs.csv")
            existing_data = responses_df

            # Append the new data to existing data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            #st.write(updated_data)
            # Save the updated data
            # For local save
            responses_df = db.update(worksheet = "Responses", data=updated_data)
            st.cache_data.clear()
            #updated_data.to_csv("human_outputs.csv", index=False)
            st.markdown(" ")
            st.markdown(" ")
            st.markdown ("Click restart to do it again!")
            finished = True
            # If using Google Sheets, replace the above line with Google Sheets API call to update the sheet
            # db.update(data=updated_data, worksheet="Sheet2")
        else:
            st.error("Data mismatch error: Something went wrong. Please restart and try again.")



        if st.button("Restart"):
            st.session_state['current_iteration'] = 0
            st.session_state['selections_list'] = []
            st.session_state['background_list'] = []
            st.session_state['experience_list'] = []
            st.session_state['age_experience'] = []
            st.session_state['time'] = []
            st.session_state['ai_prediction'] = []
            st.session_state['random_examples'] = rxn_df.sample(n=n_samples)

            st.rerun()
if finished:
    st.image(draw_results(new_data))

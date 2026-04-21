import unidecode
import re


LEGAL_PREFIX = {
    'plc', 'inc', 'company', 'tic', 'dis', 'sti', 'ltd', 'sirketi', 'ic', 'spa', 've', 'sa', 'ltdsti', 
    'san', 'as', 'pte', 'bv', 'ticaret', 'anonim', 'co', 'llc', 'gmbh', 'corp', 'ltda'
}

UNIQUE_PEOPLE_NAMES = {
    'melo', 'botelho', 'mfigueiredo', 'medeiros', 'susana', 'vasconcelos', 'alves', 'ttrovao', 'gaspar', 'carvalho', 'alberto', 'joana', 
    'brandao', 'sara', 'mateus', 'gourgel', 'lobo', 'pinho', 'biscaia', 'forreta', 'mariana', 'sergio', 'batista', 'coelho', 'ricardina', 
    'luz', 'sandra', 'baeta', 'rafael', 'canas', 'mourao', 'pila', 'joao', 'antero', 'trincheiras', 'pita', 'rasteiro', 'valerio', 'farinha', 'joel', 
    'angelo', 'aguiar', 'uva', 'alverca', 'diogo', 'abastos', 'ramos', 'mena', 'rute', 'antonio', 'novais', 'barreiros', 'madeira', 'filipa', 'jesus', 
    'paixao', 'duarte', 'lima', 'lanca', 'agostinho', 'maia', 'vinagre', 'brito', 'gil', 'fernando', 'rodrigo', 'santos', 'semiao', 'artur', 'pinto', 
    'almeida', 'alexandre', 'cunha', 'jacinto', 'vozone', 'catarina', 'olga', 'barracha', 'nuno', 'acacio', 'eduardo', 'portela', 'ernesto', 'ricardo', 
    'rilho', 'patricio', 'patricia', 'sebastiao', 'daniel', 'milene', 'nascimento', 'carlos', 'ferreira', 'figueiredo', 'renato', 'teixeira', 'o', 'carla', 
    'regateiro', 'sofia', 'isabel', 'pereira', 'dos', 'albino', 'catarino', 'sobatista', 'mendonca', 'martins', 'freitas', 'sovelas', 'peralta', 'lince', 
    'lrcorreia', 'telma', 'msantos', 'matos', 'rodrigues', 'repolholda', 'carreira', 'vaz', 'moises', 'beatriz', 'miguel', 'andre', 'jose', 
    'augusto', 'luis', 'boavida', 'moreira', 'bandeira', 'conde', 'sotero', 'casqueiro', 'engenheiro', 'caxaria', 'purificano', 'helena', 'borrego', 
    'machado', 'beirao', 'silva', 'nunes', 'edna', 'manuel', 'outros', 'santana', 'raul', 'alina', 'barbosa', 'vieira', 'travassos', 'ribeiro', 'correia', 
    'monte', 'flsmbarbosa', 'fatima', 'moutela', 'guilherme', 'mendes', 'trindade', 'pguedes', 'bruno', 'bacalhau', 'neto', 'marco', 'marques', 'leonor', 
    'goncalves', 'pedro', 'henrique', 'aduaneiro', 'armando', 'bandarra', 'antunes', 'flecha', 'delca', 'domingues', 'carmo', 'conceicao', 'abreu', 
    'fernandes', 'carminho', 'julio', 'guimaraes', 'claudia', 'severo', 'aline', 'filipe', 'horta', 'pimenta', 'bento', 'domingos', 'chagas', 'reis', 
    'guerreiro', 'baleizao', 'barros', 'fmartins', 'campos', 'barreiras', 'cavaco', 'amorim', 'faria', 'rocha', 'tania', 'crujo', 'torres', 'roberto', 
    'marta', 'branco', 'josilva', 'maria', 'fonseca', 'da', 'gomes', 'mota', 'catalao', 'militao', 'cantoneiro', 'pessoa', 'dias', 
    'nsleal', 'barreira', 'cpeixoto', 'de', 'raimundo', 'joaquim', 'do', 'fortunato', 'cascais', 'lucas', 'sousa', 'silveira', 'ines', 'despachante', 
    'mario', 'urmal', 'esteves', 'victor', 'smelro', 'simoes', 'gramatinha', 'cabral', 'e', 'oliveira', 'paulo', 'mde', 'tiago', 'lopes', 'jorge', 
    'leandro', 'ana', 'oficial', 'galhardo', 'fferreira', 'alfaiate', 'edgar', 'tavira', 'sanches', 'costa', 'monteiro', 'paiva', 'valente', 'rouxinol', 
    'amandio', 'mesquita', 'viegas', 'jacome', 'david', 'francisco', 'abel', 'azevedo', 'soares', 'elisa', 'peca', 'assis', 'fpinto', 'angeja', 'urbano', 
    'caldeira', 'simao', 'eva', 'rui', 'marrafa', 'carapinha', 'forte', 'celia', 'elisbao', 'rita', 'cruz', 'vitor', 'freire', 'garcia', 'felisbelo', 
    'raposo', 'lara'
    # 'unipessoal', 'lda', 'outro', 'filhos'
}


STOP_WORDS = {
    "rua","street","estrada","avenida","edif","largo","av"
}

TRANSLATION_TABLE = {
    r"\bldaav\b": "lda av",
    r"\bldarua\b": "lda rua",
    r"\bldaas\b": "lda as",
    r"\bto the order of\b": "",
    r"\bthe order of\b": "",
    r"\bto te order\b": "",
    r"\bto order\b": "",
    r"\bto order of\b": "",
    r"\bthe order\b": "",
    r"\bindustria e comercio\b": "",
    r"\bcomercio e industria\b": "",
    r"\bindustria\b": "",
    r"\bimportacao e exportacao\b": "",
    r"\bimportacao\b": "",
    r"\bexportacao\b": "",
    r"\bsa de cv\b": "",
    r"\brl de cv\b": "",
    r"\bas agents\b": "",
    r"\bsociedad anonima\b": "sa",
    r"\bltdsti\b": "ltd sti",
    r"\bspz oo\b": "sp z oo",
    r"\bc h\b": "ch",
    r"\bairsea\b": "air sea",
    r"\bltdaav\b": "ltda av",
    r"\btran\b": "transitarios",
    r"\btrans\b": "transitarios",
    r"\btransitariossociedade\b": "transitarios sociedade",
    r"\bj f\b": "jf",
    r"\bintl\b": "international",
    r"\binternati\b": "international",
    r"\btransitarioslda\b": "transitarios lda",
    r"\btransitrioslda\b": "transitarios lda",
    r"\btransit rios\b": "transitarios",
    r"\bldaaddress\b": "lda address",
    r"\boutroslds\b": "outros lda",
    r"\bunipessoallda\b": "unipessoal lda",
    r"\bemaritima\b": "e maritima",
    r"\bemaritimo\b": "e maritimo"

}

TRANSLATION_REGEX = [
    (re.compile(pattern), replacement)
    for pattern, replacement in sorted(TRANSLATION_TABLE.items(), key=lambda x: -len(x[0]))
]


def clean_name(name):
    name = unidecode.unidecode(name.lower())
    name = re.sub(r'[-()]', ' ', name)
    name = re.sub(r'[.,]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    for pattern, repl in TRANSLATION_REGEX:
        name = pattern.sub(repl, name)

    words = name.split()
    for index, word in enumerate(words):
        if word in STOP_WORDS and index > 0:
            words = words[:index]
            break
    name = " ".join(words)

    name = re.sub(r'[^a-z ]+', '', name)
    return re.sub(r'\s+', ' ', name).strip()


def is_person_name(name):
    return all(n in UNIQUE_PEOPLE_NAMES for n in name.split())
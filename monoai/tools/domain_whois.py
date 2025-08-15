from whois import whois

def domain_whois(domain:str):

    """
    Get the whois for a given domain:
    
    Args:
        domain (string): the domain
    """

    result = whois(domain)
    return str(result)
from whois import whois

class DomainWhois:

    def whois(self, domain: str):
        result = whois(domain)
        return result
    





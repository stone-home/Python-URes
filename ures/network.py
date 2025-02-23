import ipaddress


def verify_ip_in_subnet(ip: str, subnet: str) -> bool:
    """
    Check if a given IP address belongs to a specified subnet.

    This function converts the provided IP address and subnet (in CIDR notation)
    into their respective ipaddress objects, then checks if the IP address is within
    the network defined by the subnet.

    Args:
        ip (str): The IP address to check (e.g., "192.168.1.10").
        subnet (str): The subnet in CIDR notation (e.g., "192.168.1.0/24").

    Returns:
        bool: True if the IP address is within the subnet, False otherwise.

    Example:
        >>> verify_ip_in_subnet("192.168.1.10", "192.168.1.0/24")
        True
        >>> verify_ip_in_subnet("10.0.0.1", "192.168.1.0/24")
        False
    """
    ip_obj = ipaddress.ip_address(ip)
    network_obj = ipaddress.ip_network(subnet)
    return ip_obj in network_obj


def is_valid_ip_netmask(ip: str, netmask: str) -> bool:
    """
    Validate whether an IP address and a given subnet mask form a valid subnet.

    This function checks if the provided IP address (IPv4 or IPv6) and its subnet mask
    (provided either in CIDR notation or dot-decimal format for IPv4) are valid when used
    together to define a network.

    Args:
        ip (str): The IP address (e.g., "192.168.1.1" or "2001:db8::1").
        netmask (str): The subnet mask, which can be provided as a CIDR value (e.g., "24")
                       or in dot-decimal format (e.g., "255.255.255.0" for IPv4).

    Returns:
        bool: True if the IP address and subnet mask form a valid subnet, False otherwise.

    Example:
        >>> is_valid_ip_netmask("192.168.1.1", "255.255.255.0")
        True
        >>> is_valid_ip_netmask("192.168.1.1", "24")
        True
        >>> is_valid_ip_netmask("2001:db8::1", "64")
        True
        >>> is_valid_ip_netmask("192.168.1.1", "255.255.0.255")
        False
        >>> is_valid_ip_netmask("300.168.1.1", "24")
        False
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        if netmask.isdigit():
            prefix_length = int(netmask)
            if isinstance(ip_obj, ipaddress.IPv4Address) and 0 <= prefix_length <= 32:
                return True
            if isinstance(ip_obj, ipaddress.IPv6Address) and 0 <= prefix_length <= 128:
                return True
        else:
            try:
                # Attempt to create a network with the given dot-decimal mask.
                ipaddress.ip_network(f"{ip}/{netmask}", strict=False)
                return True
            except ValueError:
                return False
    except ValueError:
        return False


def generate_ip(subnet: str, last_index: int) -> str:
    """
    Generate a new IP address within a given subnet based on an offset.

    The function adds the provided 'last_index' to the network address of the specified
    subnet (given in CIDR notation) to compute a new IP address.

    Args:
        subnet (str): The subnet in CIDR notation (e.g., "192.168.0.0/24").
        last_index (int): The offset to add to the network address.

    Returns:
        str: The generated IP address as a string.

    Example:
        >>> generate_ip("192.168.0.0/24", 100)
        '192.168.0.100'
    """
    network = ipaddress.ip_network(subnet)
    new_ip = network.network_address + last_index
    return str(new_ip)

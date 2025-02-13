import ipaddress


def verify_ip_in_subnet(ip, subnet) -> bool:
    """Verify if an ip address is in a subnet.

    Args:
        ip (str): an ip address
        subnet (str): a subnet

    Returns:
        bool: True if the ip address is in the subnet, False otherwise
    """
    ip = ipaddress.ip_address(ip)
    subnet = ipaddress.ip_network(subnet)
    return ip in subnet


def is_valid_ip_netmask(ip: str, netmask: str) -> bool:
    """Validates whether the given IP address and network mask form a valid subnet.

    This function checks if the provided IP address (IPv4 or IPv6) and subnet mask
    (CIDR notation or dot-decimal for IPv4) form a valid combination.

    Args:
        ip (str): The IP address in string format (e.g., "192.168.1.1" or "2001:db8::1").
        netmask (str): The subnet mask, either in CIDR notation (e.g., "24") or
            dot-decimal format (e.g., "255.255.255.0" for IPv4).

    Returns:
        bool: True if the IP address and subnet mask are valid together, False otherwise.

    Examples:
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
        # Determine if it's IPv4 or IPv6
        ip_obj = ipaddress.ip_address(ip)

        if netmask.isdigit():  # If the netmask is in CIDR format (e.g., "24")
            prefix_length = int(netmask)
            if isinstance(ip_obj, ipaddress.IPv4Address) and 0 <= prefix_length <= 32:
                return True
            if isinstance(ip_obj, ipaddress.IPv6Address) and 0 <= prefix_length <= 128:
                return True
        else:  # If the netmask is in dot-decimal (e.g., "255.255.255.0")
            try:
                network = ipaddress.ip_network(f"{ip}/{netmask}", strict=False)
                return True
            except ValueError:
                return False

    except ValueError:
        return False


def generate_ip(subnet, last_index):
    """Generate an ip address based on the subnet and the last index.

    Args:
        subnet (str): a subnet
        last_index (int): the last index of the ip address.

    Examples:
        generate_ip("192.168.0.0/24", 100)
        # Output: 192.168.0.101

    Returns:
        str: a new ip address

    """
    network = ipaddress.ip_network(subnet)
    new_ip = network.network_address + last_index
    return str(new_ip)

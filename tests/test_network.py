import unittest
from ures.network import verify_ip_in_subnet, is_valid_ip_netmask, generate_ip


class TestNetworkFunctions(unittest.TestCase):

    def test_verify_ip_in_subnet_true(self):
        """Test that an IP inside the subnet returns True."""
        self.assertTrue(verify_ip_in_subnet("192.168.1.10", "192.168.1.0/24"))

    def test_verify_ip_in_subnet_false(self):
        """Test that an IP outside the subnet returns False."""
        self.assertFalse(verify_ip_in_subnet("10.0.0.1", "192.168.1.0/24"))

    def test_verify_ip_in_subnet_ipv6(self):
        """Test verify_ip_in_subnet with an IPv6 address."""
        self.assertTrue(verify_ip_in_subnet("2001:db8::1", "2001:db8::/32"))
        self.assertFalse(verify_ip_in_subnet("2001:db9::1", "2001:db8::/32"))

    def test_is_valid_ip_netmask_valid_dot_decimal(self):
        """Test is_valid_ip_netmask returns True for valid IPv4 with dot-decimal mask."""
        self.assertTrue(is_valid_ip_netmask("192.168.1.1", "255.255.255.0"))

    def test_is_valid_ip_netmask_valid_cidr(self):
        """Test is_valid_ip_netmask returns True for valid IPv4/IPv6 with CIDR mask."""
        self.assertTrue(is_valid_ip_netmask("192.168.1.1", "24"))
        self.assertTrue(is_valid_ip_netmask("2001:db8::1", "64"))

    def test_is_valid_ip_netmask_invalid(self):
        """Test is_valid_ip_netmask returns False for invalid subnet configurations."""
        self.assertFalse(is_valid_ip_netmask("192.168.1.1", "255.255.0.255"))
        self.assertFalse(is_valid_ip_netmask("300.168.1.1", "24"))

    def test_generate_ip(self):
        """Test generate_ip returns the correct IP address based on the offset."""
        result = generate_ip("192.168.0.0/24", 100)
        self.assertEqual(result, "192.168.0.100")

    def test_generate_ip_boundary(self):
        """Test generate_ip for a boundary condition."""
        result = generate_ip("10.0.0.0/8", 255)
        self.assertEqual(result, "10.0.0.255")
